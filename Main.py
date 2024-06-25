import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError  # type: ignore
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from scipy.ndimage import gaussian_filter
import argparse
import time
import os
import pickle
import json

# Register custom objects
@tf.keras.utils.register_keras_serializable()
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Define custom object scope for Mean Squared Error and Mean Absolute Error
custom_objects = {
    'mse': MeanSquaredError(),
    'mae': mae
}

# Load the trained TensorFlow model with custom object scope
model = tf.keras.models.load_model("model_detect.h5", custom_objects=custom_objects)

# Load the YOLOv7 model
yolov7_model = attempt_load('yolov7.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')

# Load the activity recognition model
activity_model = tf.keras.models.load_model('activity_model.h5')

# Load the class indices for activity recognition
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Function to preprocess the image for activity recognition
def preprocess_image(image, target_size=(150, 150)):
    img = cv2.resize(image, target_size)  # Resize the image
    img_array = tf.keras.utils.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model's input shape
    img_array = img_array / 255.0  # Rescale image
    return img_array

# Function to predict the activity
def predict_activity(model, frame, class_indices):
    img_array = preprocess_image(frame)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Map predicted class index to class label
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class[0]]
    
    return predicted_label

# Function to use YOLOv7 for object detection
def detect_people_with_yolov7(frame, model, img_size=640):
    img = letterbox(frame, img_size, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).float().cuda() if torch.cuda.is_available() else torch.from_numpy(img).float()
    img /= 255.0  # Normalize to 0-1
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[0], agnostic=False)

    boxes = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
    return boxes

# Function to process the video
def process_video(input_video_path, output_video_path, heatmap_output_path):
    if not os.path.isfile(input_video_path):
        print(f"Error: File {input_video_path} does not exist.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    target_width = 640
    target_height = int(height * (640 / width))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

    max_people_count = 0
    heatmap = np.zeros((target_height, target_width), dtype=np.float32)

    frame_data = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_width, target_height))
        boxes = detect_people_with_yolov7(frame, yolov7_model)
        people_count = len(boxes)
        max_people_count = max(max_people_count, people_count)

        activity_labels = []
        for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            person_crop = frame[y1:y2, x1:x2]
            activity_label = predict_activity(activity_model, person_crop, class_indices)
            activity_labels.append(activity_label)

            label = f'Person {idx}: {activity_label}'
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            heatmap[y1:y2, x1:x2] += 1

        frame_count_label = f'Frame Count: {people_count}'
        total_count_label = f'Total Count: {max_people_count}'
        cv2.putText(frame, frame_count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, total_count_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_data.append([cap.get(cv2.CAP_PROP_POS_FRAMES), people_count, max_people_count, timestamp, activity_labels])

        out.write(frame)

    cap.release()
    out.release()

    end_time = time.time()
    fps_processed = len(frame_data) / (end_time - start_time)
    print(f"Processed FPS: {fps_processed}")

    heatmap = gaussian_filter(heatmap, sigma=20)
    heatmap = np.uint8(255 * (heatmap / np.max(heatmap)))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_output_path, heatmap_color)

    print(f'Total people detected in the video: {max_people_count}')

    return frame_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='People Detection, Counting, and Activity Recognition in Video')
    parser.add_argument('--input', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output video file')
    parser.add_argument('--heatmap', type=str, required=True, help='Path to the output heatmap image')
    args = parser.parse_args()

    # Process video and save frame data for the dashboard and activity recognition script
    frame_data = process_video(args.input, args.output, args.heatmap)

    with open('frame_data.pkl', 'wb') as f:
        pickle.dump(frame_data, f)
