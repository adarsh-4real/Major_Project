import os
import sys
import numpy as np
import cv2
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dropout #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
import albumentations as A

# Add YOLOv7 repository to the Python path
sys.path.insert(0, 'yolov7')

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox


# Function to load and augment images from a directory with resizing
def load_and_augment_images_from_folder(folder, img_size=(640, 640)):
    images = []
    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomSizedCrop(min_max_height=(300, 640), height=640, width=640, p=0.5)
    ])
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = augment(image=img)['image']
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


# Load YOLOv7 model
def load_yolov7_model(weights_path='yolov7.pt'):
    model = attempt_load(weights_path, map_location='cpu')
    return model


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


# Load images and labels in smaller batches
image_folder = r"C:\Users\Adarsh\OneDrive\Desktop\Detect_Count\Dataset\Train\Images"
label_folder = r"C:\Users\Adarsh\OneDrive\Desktop\Detect_Count\Dataset\Train\Labels"

# Assuming batch size of 100
batch_size = 100
image_filenames = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
count_filenames = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

num_batches = len(image_filenames) // batch_size + 1

all_features = []
all_counts = []

model = load_yolov7_model()

for i in range(num_batches):
    batch_image_filenames = image_filenames[i * batch_size:(i + 1) * batch_size]
    batch_count_filenames = count_filenames[i * batch_size:(i + 1) * batch_size]

    batch_images = [cv2.imread(os.path.join(image_folder, fn)) for fn in batch_image_filenames]
    batch_images = [cv2.resize(img, (640, 640)) for img in batch_images if img is not None]
    batch_images = [A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomSizedCrop(min_max_height=(300, 640), height=640, width=640, p=0.5)
    ])(image=img)['image'] for img in batch_images]
    batch_counts = [len(open(os.path.join(label_folder, fn)).readlines()) for fn in batch_count_filenames]

    features = extract_features(batch_images, model)
    all_features.extend(features)
    all_counts.extend(batch_counts)

# Convert to NumPy arrays
all_features_np = np.array(all_features).reshape(-1, 1)
all_counts_np = np.array(all_counts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features_np, all_counts_np, test_size=0.2, random_state=42)

# Initialize and train TensorFlow regression model
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model_tf.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model_tf.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the trained TensorFlow model
model_tf.save("model_detect.h5")
