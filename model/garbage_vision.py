import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import argparse

class ImageProcessor:
    def __init__(self):
        self.model = None

    def main(self):
        parser = argparse.ArgumentParser(description="Image processing script")
        parser.add_argument('--download', action='store_true', help='Download images')
        parser.add_argument('--train', action='store_true', help='Train the model')
        parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')

        args = parser.parse_args()

        if args.download:
            self.download_images()

        if args.train:
            self.train_model()

        if args.evaluate:
            self.evaluate_model()

    def download_images(self):
        # Placeholder function for downloading images
        print("Downloading images...")

    def train_model(self):
        # Define paths
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

    def evaluate_model(self):
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
    processor = ImageProcessor()
    processor.main()
 