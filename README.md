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
