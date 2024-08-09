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


class MainApplication:
    def __init__(self):

        self.model = None
        self.train_ds , self.valid_ds = None , None

        self.image_processor = GarbageVision()
        #self.camera_stream = CameraStream()
        self.classifier = Classifier()

    def main(self):
        parser = argparse.ArgumentParser(description="Image processing script")
        parser.add_argument('--classify', action='store_true', help='Classify and create a dataset')
        parser.add_argument('--train', action='store_true', help='Train the model')
        parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
        parser.add_argument('--camera', action='store_true', help='Use camera')

        args = parser.parse_args()

        if args.classify:
            self.train_ds , self.valid_ds = self.classifier.create_dataset()

        if args.train:
            self.model = self.image_processor.create_model(2,3,2)
            self.train_model()

        if args.evaluate:
            self.image_processor.evaluate_model()

        '''if args.camera:
            self.camera_stream.show_frame()'''
    
    def train_model(self):
        
        # compile model 
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.model.summary()

        epochs = 10
        history = self.model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=epochs
        )

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


if __name__ == "__main__":
    app = MainApplication()
    app.main()
