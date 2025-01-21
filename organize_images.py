# organize_images.py

import os
import shutil


def organize_files():
    # Define the source directory where the images are currently located
    source_dir = '...'

    # Define the target directory where you want to move the organized folders
    target_dir = '....'

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        # Check if the file is an image (optional, modify as needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Extract the person number from the filename (e.g., "002A03" -> "002")
            person_number = filename[:3]

            # Define the folder name for each person
            folder_name = f'Person_{person_number}'

            # Define the path for the new folder
            new_folder_path = os.path.join(target_dir, folder_name)

            # Create the folder if it doesn't exist
            os.makedirs(new_folder_path, exist_ok=True)

            # full path for the source file
            source_file_path = os.path.join(source_dir, filename)

            # full path for the target file
            target_file_path = os.path.join(new_folder_path, filename)

            # Move the file
            shutil.move(source_file_path, target_file_path)

    print("Files have been organized successfully.")
