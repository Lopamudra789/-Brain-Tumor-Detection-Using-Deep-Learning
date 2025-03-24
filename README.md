# Brain-Tumor-Detection-Using-Deep-Learning

## Overview

Brain Tumor Detection is a deep learning-based project that utilizes a pre-trained VGG16 model to classify brain MRI images as tumorous or non-tumorous. The model is trained on a dataset of MRI scans and aims to assist radiologists in early and accurate tumor detection.

## Features

Preprocessing of MRI images (resizing, normalization, augmentation)

Deep learning model (VGG16) for tumor classification

Training and validation with performance metrics

Deployment using Flask/Streamlit for user interaction

## Technologies Used

Python

TensorFlow/Keras

OpenCV

NumPy & Pandas

Matplotlib & Seaborn

## Dataset

The dataset consists of MRI images of brain scans categorized as tumorous and non-tumorous. Data augmentation techniques are applied to enhance model performance.

## Installation

Clone the repository:

git clone https://github.com/your-repo/brain-tumor-detection.git
cd brain-tumor-detection

## Install dependencies:

pip install -r requirements.txt

Download and prepare the dataset.

## Model Training

To train the model using the VGG16 architecture, run:

python train.py

The script will train the model and save the best weights.

## Results

The trained VGG16-based model achieves high accuracy in classifying brain tumor MRI images. The evaluation metrics include precision, recall, F1-score, and confusion matrix visualization.

## Future Enhancements

Improve model accuracy with advanced architectures (ResNet, EfficientNet)

Integrate Grad-CAM for explainability

Deploy as a cloud-based web service
