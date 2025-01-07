# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 23:05:41 2025

@author: Ajay
"""

import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Define paths and parameters
data_path = "./gesture_data"  # Path to dataset folder
img_size = (64, 64)           # Resize images for consistency
num_classes = 5              # Number of gesture classes (e.g., 5 gestures)

# Load dataset
def load_dataset(data_path):
    images = []
    labels = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, img_size)
                    img = img / 255.0  # Normalize pixel values
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_dataset(data_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded, num_classes)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save the model
model.save("hand_gesture_model.h5")

# Predict on a new image
def predict_gesture(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    gesture = label_encoder.inverse_transform([np.argmax(prediction)])
    return gesture[0]

# Test prediction
new_image_path = "./test_gesture.jpg"
print("Predicted Gesture:", predict_gesture(new_image_path))