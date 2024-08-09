import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

class Classifier: 
    def __init__(self):
        self.data_path : str = r'/Users/carlosromero/Desktop/garbage_vision/dataset/training_set'

    def create_dataset(self): 
        batch_size = 32 
        img_height = 180 
        img_width = 180 
       
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_path,
            validation_split = 0.2,
            subset = 'training',
            seed = 123, 
            image_size = (img_height,img_width),
            batch_size = batch_size
        )

        valid_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_path,
            validation_split = 0.2,
            subset = 'validation',
            seed = 123, 
            image_size = (img_height,img_width),
            batch_size = batch_size
        )

        class_names = train_ds.class_names

        # visualize data in gui
        #self.visualize_data(class_names, train_ds, valid_ds)

        for image_batch , labels_batch in train_ds:
            print(image_batch.shape, labels_batch.shape)

        # stanadardize the data, rgb values are in range [0, 255] this is not ideal
        # we will standardize these values by using the range [0,1] and rescale

        normalizaion_layer = tf.keras.layers.Rescaling(1./255)

        # we can not either apply this layer within our dataset or to our model and stanadardize it.
        # We will apply it to the model 

        # we will start by configuring the dataset for performance 

        AUTOTUNE = tf.data.AUTOTUNE
        
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds , valid_ds

        '''
        Applying to data set (1 image)

        normalized_ds = train_ds.map(lambda x, y: (normalizaion_layer(x),y))
        image_batch , labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # test normalized pixels
        print(np.min(first_image), np.max(first_image))
        '''

    def visualize_data(self, class_names, train_ds, valid_ds):

        print('test', train_ds.take(1))
        plt.figure(figsize=(10,10))

        try:
            for images, labels in train_ds.take(1):
                print(f'Batch images shape: {images.shape}')
                print(f'Batch labels shape: {labels.shape}')
                
                num_images = images.shape[0]
                print(f'Number of images in the batch: {num_images}')

                for i in range(min(num_images, 9)):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i].numpy().astype('uint8'))
                    plt.title(class_names[labels[i]])
                    plt.axis('off')
            
            # Show plot even if there's an error later
            plt.show(block=True)

        except Exception as e:
            print(f'Error during visualization: {e}')





        

