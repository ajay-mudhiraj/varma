# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 23:11:39 2025

@author: Ajay
"""

import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path to dataset (assume each food item has its own folder)
data_path = "./food_data"
img_size = (128, 128)  # Standardize image size

# Dictionary to store calorie values for each food item
calorie_data = {
    "apple": 52,
    "banana": 89,
    "burger": 295,
    "pizza": 266,
    "salad": 33
}

# Load images and labels
def load_food_dataset(data_path, calorie_data):
    images = []
    labels = []
    for food_item, calorie in calorie_data.items():
        food_path = os.path.join(data_path, food_item)
        if os.path.isdir(food_path):
            for img_name in os.listdir(food_path):
                img_path = os.path.join(food_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, img_size)
                    img = img / 255.0  # Normalize pixel values
                    images.append(img)
                    labels.append(food_item)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load data
images, labels = load_food_dataset(data_path, calorie_data)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded, len(calorie_data))

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(calorie_data), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save the model
model.save("food_recognition_model.h5")

# Function to predict food item and calorie content
def predict_food(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        food_index = np.argmax(prediction)
        food_item = label_encoder.inverse_transform([food_index])[0]
        calorie_estimate = calorie_data[food_item]
        return food_item, calorie_estimate
    except Exception as e:
        print(f"Error predicting food for image {img_path}: {e}")

# Example: Predict a new image
new_image_path = "./test_food.jpg"
food_item, calorie_estimate = predict_food(new_image_path)
print(f"Predicted Food: {food_item}, Estimated Calories: {calorie_estimate} kcal")
