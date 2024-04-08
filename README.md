# Image Denoising using Machine Learning

## Introduction

This repository contains the implementation of an image denoising system using machine learning techniques, including dictionary learning and ridge regression. The project aims to investigate and implement machine learning methods for removing noise from images while preserving important features.

## Libraries Used
- Numpy
- Scikit-Learn
- Scikit-Image
- OpenCV

## How to Run:

1. Clone the repository:

    ```sh
    git clone https://github.com/akashreddy03/image-denoising.git
    cd image-denoising
    ```

2. Install necessary libraries required to run the project

3. Run the `img_capture_denoise.py` script:

    ```sh
    python img_capture_denoise.py
    ```

## Usage

1. Running the script starts video capturing.
2. Press Space to capture frames (atleast two images must be captured one for training and one for testing).
3. Press ESC to stop the video capture and start training.
4. Once the training is complete, model is applied on the test image.
5. Finally the results are displayed and also written to the same directory.
