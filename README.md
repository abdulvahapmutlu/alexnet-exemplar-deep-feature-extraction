# Pretrained CNN-based Exemplar Deep Feature Extraction
This repository contains code for extracting deep features from images using multiple pre-trained deep learning models, including AlexNet. The extracted features are then processed and selected using statistical methods for subsequent classification tasks. The code also includes functionality for handling sub-images and training a classifier.

## Features

- *Multiple Pre-trained Models*: Supports VGG16, VGG19, MobileNetV2, ResNet50, ResNet101, DenseNet201, NASNetLarge, and NASNetMobile.
- *Feature Extraction*: Extracts deep features from the penultimate or specified layers of these models.
- *Sub-image Feature Extraction*: Extracts features from sub-images for finer analysis.
- *Feature Normalization*: Normalizes features to ensure they are on a consistent scale.
- *Feature Selection*: Uses Chi-square test to select the top 1000 features.
- *Classification*: Trains a multi-class SVM classifier with a polynomial kernel.

## Code Overview

1. *Loading Models*: Loads several pre-trained models with ImageNet weights.
2. *Preprocessing*: Reads and preprocesses images according to each model's requirements.
3. *Feature Extraction*: Extracts deep learning features from the specified layers of each model.
4. *Sub-image Handling*: Extracts features from sub-images for more granular analysis.
5. *Normalization*: Normalizes the features to a common scale.
6. *Feature Selection*: Selects the top 1000 features using the Chi-square test.
7. *Classification*: Trains a multi-class SVM classifier and evaluates its performance.

## How to Use

1. *Clone the Repository*:
   sh
   git clone https://github.com/abdulvahapmutlu/pretrained-cnn-based-exemplar.git
   cd pretrained-cnn-based-exemplar
   

2. *Prepare Your Image Dataset*: Place your image files in the repository directory. Ensure the filenames contain labels and indices as required.

3. *Install Dependencies*:
   sh
   pip install tensorflow numpy scikit-learn scipy
   

4. *Run the Script*:
   sh
   python pre_cnnbased_exemplar.py
