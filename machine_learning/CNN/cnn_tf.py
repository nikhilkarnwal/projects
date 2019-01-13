#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:55:22 2019

@author: nikhilkarnwal
"""

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

class cnn_mnist:
    
    def __init__(self):
        self.batch_size = 64
        self.epochs = 20
        self.model_file = 'cnn_tf.model.best.hdf5'
    
    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        self.classes = np.max(self.y_train,axis=0)+1
        print("x_train shape:", self.x_train.shape, "y_train shape:", self.y_train.shape)
    
    def normalize(self):
        self.x_train = self.x_train.astype('float32')/255
        self.x_test = self.x_test.astype('float32')/255
        
    def split_data(self):
        (x_train, y_train) = self.x_train[5000:], self.y_train[5000:]
        (x_valid, y_valid) = self.x_train[:5000], self.y_train[:5000]
        (x_test, y_test) = self.x_test, self.y_test
        
        #change shape of image vector to 28,28,1 from 28,28
        x_train = x_train.reshape(-1,28,28,1)
        x_valid = x_valid.reshape(-1,28,28,1)
        x_test = x_test.reshape(-1,28,28,1)
        
        #convert target vector to one hot vector
        y_train = keras.utils.to_categorical(y_train,10)
        y_valid = keras.utils.to_categorical(y_valid,10)
        y_test = keras.utils.to_categorical(y_test,10)
        
        # Print training set shape
        print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

        # Print the number of training, validation, and test datasets
        print(x_train.shape[0], 'train set')
        print(x_valid.shape[0], 'validation set')
        print(x_test.shape[0], 'test set')
        
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.y_valid = x_valid, y_valid
        self.x_test, self.y_test = x_test, y_test
    
    def create_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='linear',padding='same', input_shape = (28,28,1)))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='linear', padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='linear', padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='linear'))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(self.classes, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        print(model.summary())
        self.model = model
        
    def train(self):
        checkpointer = ModelCheckpoint(self.model_file, verbose =1, save_best_only=True)
        self.model.fit(self.x_train, self.y_train, batch_size = self.batch_size, 
                       epochs = self.epochs, verbose = 1, validation_data=(self.x_valid, self.y_valid),
                       callbacks=[checkpointer])
        self.model.load_weights(self.model_file)
        
    def load_frozen_model(self, filename):
        self.model_file = filename
        self.model = keras.models.load_model(filename)
        print(self.model.summary())
        
    def evaluation(self):
        test_eval = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])
        return test_eval
    
    def prediction(self):
        y_hat = self.model.predict(self.x_test)
        
        # Plot a random sample of 10 test images, their predicted labels and ground truth
        figure = plt.figure(figsize=(20, 8))
        # Define the text labels
        fashion_mnist_labels = ["T-shirt/top",  # index 0
                                "Trouser",      # index 1
                                "Pullover",     # index 2 
                                "Dress",        # index 3 
                                "Coat",         # index 4
                                "Sandal",       # index 5
                                "Shirt",        # index 6 
                                "Sneaker",      # index 7 
                                "Bag",          # index 8 
                                "Ankle boot"]   # index 9
        for i, index in enumerate(np.random.choice(self.x_test.shape[0], size=15, replace=False)):
            ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
            # Display each image
            ax.imshow(np.squeeze(self.x_test[index]))
            predict_index = np.argmax(y_hat[index])
            true_index = np.argmax(self.y_test[index])
            # Set the title for each image
            ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                          fashion_mnist_labels[true_index]),
                                          color=("green" if predict_index == true_index else "red"))
            
    def train_and_save(self):
        self.load_data()
        self.normalize()
        self.split_data()
        self.create_model()
        self.train()
        self.evaluation()
        self.prediction()
        
    def load_and_test(self, filename):
        self.load_data()
        self.normalize()
        self.split_data()
        self.load_frozen_model(filename)
        self.evaluation()
        self.prediction()
        
cnn_test = cnn_mnist()
cnn_test.load_and_test('cnn_tf.model.best.hdf5')