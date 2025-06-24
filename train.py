import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils import get_data_generators

DATA_DIR = "data_split"
MODEL_PATH = "models/fake_image_model.h5"
BATCH_SIZE = 32
EPOCHS = 30
IMG_SHAPE = (224, 224, 3)

def build_model():
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=IMG_SHAPE)
    base_model.trainable = True  # Fine-tune all layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    return model

def get_class_weights(train_gen):
    # Compute class weights based on training data distribution
    labels = train_gen.classes
    class_totals = np.bincount(labels)
    total = float(sum(class_totals))
    class_weight = {i: total / (2.0 * class_totals[i]) for i in range(len(class_totals))}
    return class_weight

def train():
    os.makedirs("models", exist_ok=True)
    train_gen, val_gen = get_data_generators(DATA_DIR, batch_size=BATCH_SIZE)

    model = build_model()

    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    class_weight = get_class_weights(train_gen)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )

    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
