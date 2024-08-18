import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import argparse
import matplotlib.pyplot as plt 
import numpy as np 


class GarbageVision:
    def __init__(self):
        self.model = None

    def create_sequential_model(self, num_classes : int, num_blocks : int, num_layers : int, drop_val : float) -> tf.keras.Sequential: 
        # create sequential model and rescale 
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Rescaling(1./255))
   
        # add convolutional blocks, dense layers and output layer
        for _ in range(num_blocks):
            self.model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
            self.model.add(tf.keras.layers.MaxPooling2D())

        # flatten the output from the convolutional blocks 
        self.model.add(tf.keras.layers.Flatten())
        
        for _ in range(num_layers):
            self.model.add(tf.keras.layers.Dense(128, activation='relu'))

        if drop_val is None:
            self.model.add(tf.keras.layers.Dense(num_classes))
        else:
            for _ in range(num_blocks):
                tf.keras.layers.Dropout(drop_val)

            self.model.add(tf.keras.layers.Dense(num_classes))
            tf.keras.layers.Dropout(drop_val)

        return self.model 

    def evaluate_model(self , epochs : int, history) -> int:

        # logs for visualization
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        self.model.summary()

    def predict(self, class_names) -> None:
        predict_data_path : str = r'/Users/carlosromero/Desktop/garbage_vision/prediction_data/'

        for filename in os.listdir(predict_data_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                file_path = os.path.join(predict_data_path, filename)
                img = tf.keras.utils.load_img(
                    file_path, target_size=(180, 180)
                )
                
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create a batch

                predictions = self.model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                print(
                    f"This image most likely belongs to {class_names[np.argmax(score)]} "
                    f"with a {100 * np.max(score):.2f} percent confidence."
                )
    
    def convert_model(self):
        # convert model 
        converter = tf.lite.TFLiteConverter.from_kera_model(model)
        tflite_model = converter.convert()
            
    def train_model_transfer(self):
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
 