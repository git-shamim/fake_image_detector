import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# Set the image size expected by the model
IMG_SIZE = (224, 224)

def get_data_generators(base_dir="data_split", batch_size=32):
    # Training data augmentation
    datagen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30.0
    )

    # Validation generator: only rescaling
    datagen_val = ImageDataGenerator(rescale=1. / 255)

    train_gen = datagen_train.flow_from_directory(
        os.path.join(base_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_gen = datagen_val.flow_from_directory(
        os.path.join(base_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen

def load_and_prepare_image(path):
    """
    Load an image from disk, resize to IMG_SIZE, and prepare for model prediction.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Unable to load image at {path}")
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
