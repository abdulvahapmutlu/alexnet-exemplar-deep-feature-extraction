import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import (
    VGG16, VGG19, MobileNetV2, ResNet50, ResNet101, DenseNet201, NASNetLarge, NASNetMobile
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import scipy.io as sio

# Define models and corresponding preprocess functions
models = [
    VGG16(weights='imagenet'),
    VGG19(weights='imagenet'),
    MobileNetV2(weights='imagenet'),
    ResNet50(weights='imagenet'),
    ResNet101(weights='imagenet'),
    DenseNet201(weights='imagenet'),
    NASNetLarge(weights='imagenet'),
    NASNetMobile(weights='imagenet')
]

layer_names = [
    'fc2', 'fc2', 'Logits', 'avg_pool', 'avg_pool', 'avg_pool', 'predictions', 'predictions'
]

preprocess_funcs = [
    vgg_preprocess, vgg_preprocess, mobilenet_preprocess,
    resnet_preprocess, resnet_preprocess, resnet_preprocess,
    resnet_preprocess, resnet_preprocess
]

# Initialize lists to store results
X = []
y = []
indices = []

# Iterate over each model
for model, layer_name, preprocess_func in zip(models, layer_names, preprocess_funcs):
    # Get the input size of the model
    input_size = model.input_shape[1:3]

    # Iterate through all image files in the directory
    for filename in os.listdir('.'):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Read and preprocess the image
            img = image.load_img(filename, target_size=input_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_func(img_array)

            # Extract features using the specified layer
            feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            features = feature_model.predict(img_array)

            # Append the features and labels to the lists
            X.append(features.flatten())
            y.append(int(filename[0]))
            indices.append(int(filename[0:2]))  # Adjust this as per your filename format

            # Extract sub-images and their features
            sub_images = []
            step_size = img_array.shape[1] // 3
            for i in range(0, img_array.shape[1], step_size):
                for j in range(0, img_array.shape[2], step_size):
                    sub_img = img_array[:, i:i + step_size, j:j + step_size, :]
                    if sub_img.shape[1:3] == (step_size, step_size):
                        sub_features = feature_model.predict(sub_img)
                        X.append(sub_features.flatten())

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)
indices = np.array(indices)

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Feature selection using Chi-square test
selector = SelectKBest(score_func=chi2, k=1000)
X_new = selector.fit_transform(X, y)

# Train an SVM classifier with polynomial kernel
svm = OneVsRestClassifier(SVC(kernel='poly', degree=3, C=1, probability=True))
scores = cross_val_score(svm, X_new, y, cv=10)
error = 1 - scores.mean()

