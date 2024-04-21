from unittest import result
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new layers for binary classification
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout for regularization
predictions = Dense(2, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

def load_and_preprocess_image(img_path):
    """
    Loads an image file and preprocesses it for model prediction.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def predict_image(model, img_path):
    """
    Predict whether an image is real or fake.
    """
    processed_img = load_and_preprocess_image(img_path)
    predictions = model.predict(processed_img)
    return "Fake" if np.argmax(predictions) == 0 else "Real"

# Example Usage:
# Assuming you have a model trained or you're using this structure hypothetically.
img_path = '/home/snorpiii/Pictures/photo_2023-02-28_16-54-13.jpg'
# result = predict_image(model, img_path)
print("The image is:", predictions)
