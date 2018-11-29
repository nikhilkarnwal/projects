import keras
from keras.models import Sequential,Input,Model,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

import sys
from sklearn.model_selection import train_test_split

#load and preprocess data
(tn_x, tn_y), (tt_x, tt_y) = fashion_mnist.load_data()
valid_x, valid_y = tn_x, tn_y


def train_model_func():
    batch_size = 64
    epochs = 20
    num_classes = 10

    #model architecture

    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))

    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.4))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dropout(0.3))
    fashion_model.add(Dense(num_classes, activation='softmax'))

    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    print(fashion_model.summary())



    ######  Train the model #####
    fashion_model.fit(tn_x, tn_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_x, valid_y))

    print('Training done!')
    return fashion_model;

def preprocess_data():
    global tn_x, tt_x, tn_y, tt_y, valid_x, valid_y
    plt.figure(figsize=[5, 5])
    plt.subplot(121)
    plt.imshow(tn_x[0,:,:], cmap='gray')
    plt.title('Ground Truth : {}'.format(tn_y[0]))

    ##Data Preprocessing

    tn_x = tn_x.reshape(-1, 28, 28, 1).astype('float32')/255
    tt_x = tt_x.reshape(-1, 28, 28, 1).astype('float32')/255
    
    ##change labels to one-hot encoding
    tn_y_one_hot = to_categorical(tn_y)
    # Display the change for category label using one-hot encoding
    #print('Original label:', tn_y[0])
    #print('After conversion to one-hot:', tn_y_one_hot[0])
    tn_x, valid_x, tn_y, valid_y = train_test_split(tn_x, tn_y_one_hot, test_size=0.2, random_state=13)
    #print(tn_x.shape, tt_x.shape)
    print('Data preprocessed')


def load_model_func(model_name):
    fashion_model = load_model(model_name)
    return fashion_model

def save_model_func(fashion_model, model_name):
    fashion_model.save(model_name)

def evaluation(fashion_model):
    tt_y_one_hot = to_categorical(tt_y)
    test_eval = fashion_model.evaluate(tt_x, tt_y_one_hot, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

def prediction(fashion_model):
    predicted_classes = fashion_model.predict(tt_x)
    predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
    print('Found %d correct labels', len(np.where(predicted_classes==tt_y)[0]))


if __name__ == '__main__':
    preprocess_data()
    model_name = 'cnn_model.h5'
    if len(sys.argv) == 1:
        model = train_model_func()
        save_model_func(model, model_name)
    else:
        model = load_model_func(model_name)

    evaluation(model)
    prediction(model)




