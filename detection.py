import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Filter out CUDA-related warnings
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam # type: ignore

# Function to load images and labels with error handling
def load_images_and_labels(image_dirs, img_size=(160, 160)):
    images, labels = [], []
    for label, image_dir in enumerate(image_dirs):
        for filename in os.listdir(image_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(image_dir, filename)
            try:
                img = load_img(img_path, target_size=img_size)
                images.append(img_to_array(img) / 255.0)
                labels.append(label)
            except IOError as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Directories for real and fake images
real_images_dir = '/kaggle/input/casia-dataset/CASIA2/Au'
fake_images_dir = '/kaggle/input/casia-dataset/CASIA2/Tp'

# Load dataset and split
images, labels = load_images_and_labels([real_images_dir, fake_images_dir])
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=25, validation_data=(test_images, test_labels), verbose=1)

