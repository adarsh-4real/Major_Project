import os
import shutil

def rename_and_collect_images(input_folder, output_folder):
    """
    Recursively rename .jpg files in the input folder and subfolders, saving them to a common output folder.
    The renaming starts from 1.jpg and continues sequentially through all images.

    Parameters:
    - input_folder: str, path to the folder containing the input .jpg files.
    - output_folder: str, path to the folder where renamed images will be saved.
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    current_index = 1

    for root, dirs, files in os.walk(input_folder):
        # Filter and sort .jpg files
        jpg_files = sorted([f for f in files if f.endswith('.jpg')])

        for filename in jpg_files:
            # Construct the new filename
            new_name = f"{current_index}.jpg"

            # Full path to the old and new files
            old_file = os.path.join(root, filename)
            new_file = os.path.join(output_folder, new_name)

            # Copy and rename the file to the output folder
            shutil.copy2(old_file, new_file)
            print(f"Copied and renamed '{old_file}' to '{new_file}'")

            current_index += 1

# Example usage
input_folder = r"C:\Users\Adarsh\OneDrive\Desktop\Activity_Dataset\Activity_Dataset\Sub"
output_folder = r"C:\Users\Adarsh\OneDrive\Desktop\Activity_Dataset\Activity_Dataset\Smiling"
rename_and_collect_images(input_folder, output_folder)
