# Vision Transformer-Enhanced Multi-Descriptor Ap-proach for Robust Age-Invariant Face Recognition

This repository contains Python scripts for processing images from the MORPH II &FGNET datasets, converting them to grayscale, detecting faces, resizing them, and extracting features using ViT (Vision Transformer), SIFT (Scale-Invariant Feature Transform), and LBP (Local Binary Patterns). The extracted features are then fused and dimensions reduced for classification with machine learning models such as Random Forest and K-Nearest Neighbors.

## Features

- Converts all images in a folder to grayscale and saves them in an output folder.
- Detects faces in images, crops them, resizes them to 224x224 pixels, and saves them in a separate folder.
- Extracts features from images using SIFT, LBP, and ViT (Vision Transformer).
- Combines features into a fixed-length vector and prepares them for classification.
- Supports the processing of multiple images in a directory structure while maintaining the directory hierarchy.

## Requirements

- Python 3.6+
- OpenCV
- scikit-learn
- Transformers (HuggingFace)
- PyTorch
- NumPy
- Matplotlib
- tqdm
- seaborn

You can install the required dependencies using:

```bash
pip install -r requirements.txt

## Usage

1. **Convert Images to Grayscale**  
   The script converts all images in the specified input folder to grayscale and saves them in the output folder while maintaining the directory structure.
   
   ```python
   convert_images_to_grayscale(input_folder_path, output_folder_path)

2. **Face Extraction and Resizing**
    The script detects faces in the grayscale images, extracts them, resizes them to 224x224 pixels, and saves them in a new directory.

    ```python
    process_all_images(output_folder_path, cropped_folder_path)


