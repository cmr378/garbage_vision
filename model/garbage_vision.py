"""Utilities for training and evaluating simple garbage vision models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class GarbageVision:
    """High level helper for constructing and evaluating vision models."""

    def __init__(self) -> None:
        # ``self.model`` will be populated by the builder methods.
        self.model: tf.keras.Model | None = None

    def create_sequential_model(
        self,
        num_classes: int,
        num_blocks: int,
        num_layers: int,
        drop_val: float | None,
    ) -> tf.keras.Sequential:
        """Create a simple convolutional network.

        Parameters
        ----------
        num_classes:
            Number of classes to predict.
        num_blocks:
            How many convolution/max‑pool blocks to add.
        num_layers:
            Number of dense layers after the convolutional part.
        drop_val:
            Optional dropout rate applied before the output layer.
        """

        model = tf.keras.Sequential([tf.keras.layers.Rescaling(1.0 / 255)])

        for _ in range(num_blocks):
            model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
            model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Flatten())

        for _ in range(num_layers):
            model.add(tf.keras.layers.Dense(128, activation="relu"))

        if drop_val is not None:
            model.add(Dropout(drop_val))

        activation = "softmax" if num_classes > 1 else "sigmoid"
        model.add(tf.keras.layers.Dense(num_classes, activation=activation))

        self.model = model
        return model

    def plot_training_history(self, epochs: int, history) -> None:
        """Visualise accuracy and loss for the given training history."""

        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

        if self.model is not None:
            self.model.summary()

    def predict(self, class_names, predict_data_path: str = "prediction_data") -> None:
        """Run predictions for every image in ``predict_data_path``."""

        path = Path(predict_data_path)
        for file_path in path.iterdir():
            if file_path.suffix.lower() not in {".jpg", ".png"}:
                continue

            img = tf.keras.utils.load_img(file_path, target_size=(180, 180))
            img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)

            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            print(
                f"This image most likely belongs to {class_names[np.argmax(score)]} "
                f"with a {100 * np.max(score):.2f} percent confidence."
            )

    def convert_model(self, output_file: str = "model.tflite") -> bytes:
        """Convert the current model to TensorFlow Lite format."""

        if self.model is None:
            raise ValueError("No model available to convert.")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(output_file, "wb") as f:
            f.write(tflite_model)
        return tflite_model
            
    def train_model_transfer(self) -> None:
        """Train a binary classifier using a VGG16 base model."""

        train_dir = 'dataset/training_set'
        validation_dir = 'dataset/validation_set'

        # Data augmentation and rescaling for the training set
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.8, 1.2]
        )

        # Rescaling for the validation set
        validation_datagen = ImageDataGenerator(rescale=1./255)

        # Load and augment the training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary'
        )

        # Load the validation data
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary'
        )

        # Load pre-trained VGG16 model + higher level layers
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

        # Freeze convolutional layers
        for layer in base_model.layers:
            layer.trainable = False

        # Create top model
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)

        # This is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=50,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[early_stopping]
        )

        # Save the model
        self.model.save('model.h5')

    def evaluate_model(self) -> None:
        """Evaluate the stored model using the validation set."""

        if self.model is None:
            self.model = tf.keras.models.load_model('model.h5')

        # Define path for validation set
        validation_dir = 'dataset/validation_set'

        # Rescaling for the validation set
        validation_datagen = ImageDataGenerator(rescale=1./255)

        # Load the validation data
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary'
        )

        # Evaluate the model
        loss, accuracy = self.model.evaluate(validation_generator)
        print(f'Validation Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Garbage Vision utility")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="build a dummy model and print its summary",
    )
    args = parser.parse_args()

    if args.summary:
        gv = GarbageVision()
        gv.create_sequential_model(num_classes=2, num_blocks=1, num_layers=1, drop_val=0.5)
        gv.model.summary()
 