import os
import numpy as np
import cv2
import torch
import tensorflow as tf
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to load images from a directory with resizing
def load_images_from_folder(folder, img_size=(640, 640)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = letterbox(img, img_size, stride=32)[0]  # Use letterbox to maintain aspect ratio
                images.append(img)
    return images

# Function to load counts from a directory (assuming each line is a bounding box count)
def load_counts_from_folder(folder):
    counts = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(folder, filename)
            with open(txt_path, 'r') as file:
                count = len(file.readlines())  # Counting the number of lines (each representing a bounding box)
                counts.append(count)
    return counts

# Function to use YOLOv7 for object detection (using pre-trained weights)
def detect_people_with_yolov7(frame, model, img_size=640):
    img = letterbox(frame, img_size, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).float()
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
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
    return boxes

# Function to extract features based on YOLOv7 detections
def extract_features(images, model):
    features = []
    for image in images:
        boxes = detect_people_with_yolov7(image, model)
        feature = len(boxes)  # Example: count of bounding boxes
        features.append(feature)
    return np.array(features).reshape(-1, 1)  # Reshape for model training

# Load YOLOv7 model
def load_yolov7_model(weights_path='yolov7.pt'):
    model = attempt_load(weights_path, map_location='cpu')
    return model

# Load the trained TensorFlow model with custom objects
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
tf_model = tf.keras.models.load_model("model_detect.h5", custom_objects=custom_objects)

# Load YOLOv7 model
yolov7_model = load_yolov7_model()

# Load validation images and labels
val_image_folder = r"C:\Users\Adarsh\OneDrive\Desktop\Detect_Count\Dataset\Valid\Images"
val_label_folder = r"C:\Users\Adarsh\OneDrive\Desktop\Detect_Count\Dataset\Valid\Labels"

val_images = load_images_from_folder(val_image_folder)
val_counts = load_counts_from_folder(val_label_folder)

# Extract features for validation data
val_features = extract_features(val_images, yolov7_model)

# Predict using the trained TensorFlow model
val_predictions = tf_model.predict(val_features)

# Calculate evaluation metrics
mae = mean_absolute_error(val_counts, val_predictions)
mse = mean_squared_error(val_counts, val_predictions)
r2 = r2_score(val_counts, val_predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((val_counts - val_predictions) / val_counts)) * 100

print(f'Validation MAE: {mae:.2f}')
print(f'Validation MSE: {mse:.2f}')
print(f'Validation R² Score: {r2:.2f}')
print(f'Validation RMSE: {rmse:.2f}')

