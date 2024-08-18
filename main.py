import argparse
import tensorflow as tf
from functions.camera_stream import CameraStream
from functions.classifier import Classifier
from model.garbage_vision import GarbageVision
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os 


class MainApplication:
    def __init__(self):

        model , history , model_path = None , None , '/garbage_vision.keras'
        train_ds , valid_ds , class_names = None , None , None

        image_processor = GarbageVision()
        classifier = Classifier()

        # check if a model exists:
        if os.path.isdir('garbage_vision.keras'):
            model = keras.models.load_model('/garbage_vision.keras')
        else:
            # create dataset with images in our directory 
            train_ds , valid_ds , class_names = classifier.create_dataset()

            # create a sequential model for 2 classes, 3 conv blocks, 2 pooling layers, and drop val
            model = image_processor.create_sequential_model(num_classes=2,num_blocks=3,num_layers=2,drop_val=.2)
            # train the model 
            self.train_model(model, train_ds, valid_ds)

            image_processor.predict(class_names)
            model.save('garbage_vision.keras')

            
    def train_model(self, model, train_ds, valid_ds):
        
        # compile model 
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        epochs = 20
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=epochs
        )

        return history

if __name__ == "__main__":
    app = MainApplication()