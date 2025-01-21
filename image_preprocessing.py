import os
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define constants
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


# ----------------------
# Grayscale Conversion
# ----------------------
def convert_images_to_grayscale(input_folder, output_folder):
    """Convert images in the input folder to grayscale and save them in the output folder."""
    os.makedirs(output_folder, exist_ok=True)

    total_images = sum(
        len(files) for r, d, files in os.walk(input_folder) if
        any(file.lower().endswith(SUPPORTED_FORMATS) for file in files)
    )
    processed_images = 0
    processed_image_paths = []

    for subdir, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.lower().endswith(SUPPORTED_FORMATS):
                file_path = os.path.join(subdir, file_name)
                relative_path = os.path.relpath(subdir, input_folder)
                dest_subdir = os.path.join(output_folder, relative_path)

                os.makedirs(dest_subdir, exist_ok=True)

                image = cv2.imread(file_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                gray_file_path = os.path.join(dest_subdir, file_name)
                cv2.imwrite(gray_file_path, gray_image)

                processed_image_paths.append(file_path)
                processed_images += 1
                print(f"Processed {processed_images}/{total_images} images", end='\r')

    print("\nConversion to grayscale completed.")
    _display_random_image(processed_image_paths)


# Helper function for displaying random images
def _display_random_image(image_paths):
    """Display a random image from the list of image paths."""
    if image_paths:
        random_file_path = random.choice(image_paths)
        random_image = cv2.imread(random_file_path)
        random_gray_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2GRAY)

        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Grayscale Image')
        plt.imshow(random_gray_image, cmap='gray')
        plt.axis('off')

        plt.show()
    else:
        print("No images found to display.")


# ------------------
# Face Cropping
# ------------------
def extract_and_resize_faces(filename, face_cascade, source_dir, destination_dir):
    """Extract faces from an image, resize them, and save them."""
    data = cv2.imread(filename)
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    source_subfolder = os.path.relpath(os.path.dirname(filename), source_dir)
    dest_subdir = os.path.join(destination_dir, source_subfolder)
    os.makedirs(dest_subdir, exist_ok=True)

    for i, (x, y, w, h) in enumerate(faces):
        face = data[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (224, 224))
        face_filename = os.path.join(dest_subdir, f"{os.path.splitext(os.path.basename(filename))[0]}_{i}.jpg")
        cv2.imwrite(face_filename, resized_face)


def process_all_images_for_faces(source_dir, destination_dir):
    """Process all images in the source directory to extract and save faces."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    os.makedirs(destination_dir, exist_ok=True)

    image_count = 0
    for subdir, _, files in os.walk(source_dir):
        for file_name in files:
            if file_name.lower().endswith(SUPPORTED_FORMATS):
                file_path = os.path.join(subdir, file_name)
                extract_and_resize_faces(file_path, face_cascade, source_dir, destination_dir)
                image_count += 1

    print(f"Processed {image_count} images.")


def display_random_image_with_faces(source_dir):
    """Display a random image with detected face bounding boxes."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_files = [
        os.path.join(subdir, file_name)
        for subdir, _, files in os.walk(source_dir)
        for file_name in files
        if file_name.lower().endswith(SUPPORTED_FORMATS)
    ]

    if not image_files:
        print("No images found in the source directory.")
        return

    random_file = random.choice(image_files)
    data = cv2.imread(random_file)
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    data_with_boxes = data.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(data_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(data_with_boxes, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image with Detected Faces")
    axes[1].axis('off')

    plt.show()