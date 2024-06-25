import cv2
import numpy as np
import torch
import tensorflow as tf
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from letterbox import letterbox
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.ndimage import gaussian_filter
import time
import pickle
import json

# Define custom object scope for Mean Squared Error and Mean Absolute Error
@tf.keras.utils.register_keras_serializable()
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'mae': mae
}

# Load the YOLOv7 model
yolov7_model = attempt_load('yolov7.pt', map_location='cpu').autoshape()

# Load the custom TensorFlow models
detect_model = tf.keras.models.load_model("model_detect.h5", custom_objects=custom_objects)
activity_model = tf.keras.models.load_model('activity_model.h5')

# Load the class indices for activity recognition
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Initialize the Deep SORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)

# Function to preprocess the image for activity recognition
def preprocess_image(image, target_size=(150, 150)):
    img = cv2.resize(image, target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict the activity
def predict_activity(model, frame, class_indices):
    img_array = preprocess_image(frame)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class[0]]
    return predicted_label

# Function to use YOLOv7 for object detection
def detect_people_with_yolov7(frame, model, img_size=640):
    img, _, _, _ = letterbox(frame, new_shape=(img_size, img_size))
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
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
                boxes.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf)))
    return boxes

# Function to process the video stream in real-time
def process_video_realtime(video_stream, output_path=r"C:\Users\Adarsh\OneDrive\Desktop\Detect_Count\yolov7\rel_vid\output_video.avi", heatmap_output_path="heatmap.png"):
    max_people_count = 0
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_stream.get(cv2.CAP_PROP_FPS))
    target_width = 640
    target_height = int(frame_height * (640 / frame_width))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    heatmap = np.zeros((target_height, target_width), dtype=np.float32)
    frame_data = []
    start_time = time.time()

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_width, target_height))
        boxes = detect_people_with_yolov7(frame, yolov7_model)
        people_count = len(boxes)
        max_people_count = max(max_people_count, people_count)

        activity_labels = []
        for idx, (x1, y1, x2, y2, conf) in enumerate(boxes, start=1):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            activity_label = predict_activity(activity_model, person_crop, class_indices)
            activity_labels.append(activity_label)
            label = f'Person {idx}: {activity_label}'
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - label_height - baseline), (int(x1) + label_width, int(y1)), (0, 255, 0), -1)
            cv2.putText(frame, label, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            heatmap[int(y1):int(y2), int(x1):int(x2)] += 1

        frame_count_label = f'Frame Count: {people_count}'
        total_count_label = f'Total Count: {max_people_count}'
        cv2.putText(frame, frame_count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, total_count_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        timestamp = video_stream.get(cv2.CAP_PROP_POS_MSEC)
        frame_data.append([video_stream.get(cv2.CAP_PROP_POS_FRAMES), people_count, max_people_count, timestamp, activity_labels])
        out.write(frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    fps_processed = len(frame_data) / (end_time - start_time)
    print(f"Processed FPS: {fps_processed}")

    heatmap = gaussian_filter(heatmap, sigma=20)
    heatmap = np.uint8(255 * (heatmap / np.max(heatmap)))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_output_path, heatmap_color)

    print(f'Total people detected in the video: {max_people_count}')

    with open('frame_data_rl.pkl', 'wb') as f:
        pickle.dump(frame_data, f)

# Example usage
if __name__ == "__main__":
    input_video_path = 0  # Use 0 for webcam, or replace with the path to your video file
    video_stream = cv2.VideoCapture(input_video_path)
    process_video_realtime(video_stream)
