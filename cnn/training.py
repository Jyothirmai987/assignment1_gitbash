import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import models,layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from google.colab import files
from tensorflow.keras.models import load_model
import json
from types import SimpleNamespace


# pre-processing and training the model 
class CNN:
    def pre_processing(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()

        # normalize it to 0-1
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255
        # converting 2d to 1d
        # Assuming grayscale images
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28, 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28, 1))

    def cnn_model(self):
        self.model = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    def load_config(self,path):
      with open(path,'r') as f:
        data=json.load(f)
      self.config_obj=SimpleNamespace(**data)




    def Training_model(self):
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config_obj.path,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
        self.model.compile(optimizer=self.config_obj.optimizer, loss=self.config_obj.loss, metrics=self.config_obj.metrics)
        self.model.fit(self.x_train, self.y_train, epochs=int(self.config_obj.epochs), validation_data=(self.x_test, self.y_test),callbacks=[self.model_checkpoint_callback])
        self.model.evaluate(self.x_test, self.y_test)

        # Save the trained model
        self.model.save(self.config_obj.model)
obj = CNN()
obj.pre_processing()
obj.cnn_model()
obj.load_config('C:\Users\Mantra\Desktop\learning\assignment1\config.json')
obj.Training_model()
