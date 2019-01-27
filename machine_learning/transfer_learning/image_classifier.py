"""
@author: nikhilkarnwal
"""

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from dataset import DataSet


class ImageClassifier:
    data_set = DataSet()

    def __init__(self, model_name, _dataset):
        self.batch_size = 100
        self.epochs = 4000
        self.model_file = model_name
        self.data_set = _dataset

    def load_data(self):
        (self.x_train, self.y_train), (self.x_valid, self.y_valid), (
            self.x_test, self.y_test) = self.data_set.load_data()
        self.classes = len(self.data_set.category)
        print("x_train shape:", self.x_train.shape, "y_train shape:",
              self.y_train.shape)

    def normalize(self):
        self.x_train = self.x_train.astype('float32') / 255
        self.x_valid = self.x_valid.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

    def split_data(self):

        #convert target vector to one hot vector
        y_train = keras.utils.to_categorical(self.y_train, self.classes)
        y_valid = keras.utils.to_categorical(self.y_valid, self.classes)
        y_test = keras.utils.to_categorical(self.y_test, self.classes)
        #print(y_valid)
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test

    def create_model(self):
        model = keras.Sequential()
        model.add(
            keras.layers.Dense(
                self.classes,
                activation='softmax',
                input_shape=self.data_set.dim))

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def train(self):
        checkpointer = ModelCheckpoint(
            self.model_file, verbose=0, save_best_only=True)
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
            validation_data=(self.x_valid, self.y_valid),
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
        fashion_mnist_labels = [k for k in self.data_set.cat_to_id.keys()]
        for i, index in enumerate(
                np.random.choice(self.x_test.shape[0], size=15,
                                 replace=False)):
            ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
            # Display each image
            #img_io.imshow(self.x_test[index])
            #img_io.show()
            print(y_hat[index])
            ax.imshow((self.x_test[index] * 255))
            predict_index = np.argmax(y_hat[index])
            true_index = np.argmax(self.y_test[index])
            # Set the title for each image
            ax.set_title(
                "{} ({})".format(fashion_mnist_labels[predict_index],
                                 fashion_mnist_labels[true_index]),
                color=("green" if predict_index == true_index else "red"))

    #use normalize and prediction function if input
    #is image instead of feature vector
    def train_and_save(self):
        self.load_data()
        #self.normalize()
        self.split_data()
        self.create_model()
        self.train()
        self.evaluation()
        #self.prediction()

    def load_and_test(self, filename):
        self.load_data()
        #self.normalize()
        self.split_data()
        self.load_frozen_model(filename)
        self.evaluation()
        #self.prediction()
