import os
import json

# Path to your images
image_folder = r'C:\Users\Adarsh\OneDrive\Desktop\Activity_Dataset\Activity_Dataset'

# Get all subdirectories (each subdirectory represents a class)
classes = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

# Map each class to an integer label
class_to_label = {cls: idx for idx, cls in enumerate(classes)}

# Example: Assign labels to all images in the directory and save to JSON
labels = []
for cls in classes:
    class_folder = os.path.join(image_folder, cls)
    for img_file in os.listdir(class_folder):
        if img_file.endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(class_folder, img_file)
            labels.append({'Image_Path': img_path, 'Label': class_to_label[cls]})

# Save labels to JSON file
json_filename = 'image_labels.json'
with open(json_filename, 'w') as jsonfile:
    json.dump(labels, jsonfile, indent=4)

print(f"Labels saved to {json_filename}")
