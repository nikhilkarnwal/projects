# Image Classification using CNNs and tensorflow library
This tutorial briefly talk about the implementation of Convolutional Neural Networks using tensorflow’s keras library. 
I have developed an Image Classifier using CNNs for classifying mnsit fashion dataset.

File <B>cnn_tf.py</B> contain code for loading mnsit_fashion data, training on it and testing it. <B>create_model(self)</B> function contain the code to create model and compile it as provided below. 

    class cnn_mnist:
      def __init__(self):
          self.batch_size = 64
          self.epochs = 20
          self.model_file = 'cnn_tf.model.best.hdf5'

      ..........
      ..........

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
          .......

Function <B> train_and_save() </B> will initialize, load data, create model, train the model and save it for directly using it for prediction as followed - 

    def train_and_save(self):
        self.load_data()
        self.normalize()
        self.split_data()
        self.create_model()
        self.train()
        self.evaluation()
        self.prediction()

Function <B> load_and_test(model_file) </B> will load test data and load model save in previous step to run it on test data as followed. It can be modified to test the model on new set of images by processinng images to (28,28,1).

    def load_and_test(self, filename):
        self.load_data()
        self.normalize()
        self.split_data()
        self.load_frozen_model(filename)
        self.evaluation()
        self.prediction()
        
Import cnn_mnsit module and train/test it using following code - 

    from cnn_tf import cnn_mnsit
    cnn_test = cnn_mnist()
    cnn_test.train_and_save()
    model_file_name = cnn_test.model_file
    cnn_test.load_and_test(model_file_name)

# Training Results 
    Summary of model -

    x_train shape: (60000, 28, 28) y_train shape: (60000,)
     x_train shape: (55000, 28, 28, 1) y_train shape: (55000, 10)
     55000 train set
     5000 validation set
     10000 test set

     Layer (type)                 Output Shape              Param #   
     conv2d_6 (Conv2D)            (None, 28, 28, 32)        320       

     leaky_re_lu_8 (LeakyReLU)    (None, 28, 28, 32)        0         

     max_pooling2d_6 (MaxPooling2 (None, 14, 14, 32)        0         

     dropout_8 (Dropout)          (None, 14, 14, 32)        0         

     conv2d_7 (Conv2D)            (None, 14, 14, 64)        18496     

     leaky_re_lu_9 (LeakyReLU)    (None, 14, 14, 64)        0         

     max_pooling2d_7 (MaxPooling2 (None, 7, 7, 64)          0         

     dropout_9 (Dropout)          (None, 7, 7, 64)          0         

     conv2d_8 (Conv2D)            (None, 7, 7, 128)         73856     

     leaky_re_lu_10 (LeakyReLU)   (None, 7, 7, 128)         0         

     max_pooling2d_8 (MaxPooling2 (None, 4, 4, 128)         0         

     dropout_10 (Dropout)         (None, 4, 4, 128)         0         

     flatten_2 (Flatten)          (None, 2048)              0         

     dense_4 (Dense)              (None, 128)               262272    

     leaky_re_lu_11 (LeakyReLU)   (None, 128)               0         

     dropout_11 (Dropout)         (None, 128)               0         

     dense_5 (Dense)              (None, 10)                1290      
     Total params: 356,234
     Trainable params: 356,234
     Non-trainable params: 0
     
     Train on 55000 samples, validate on 5000 samples
     Epoch 1/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.5747 - acc: 0.7886  
     Epoch 00001: val_loss improved from inf to 0.34005, saving model to cnn_tf.model.best.hdf5
     55000/55000 [==============================] - 71s 1ms/step - loss: 0.5747 - acc: 0.7886 - val_loss: 0.3400 - val_acc: 0.8788
     Epoch 2/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.3650 - acc: 0.8660  
     Epoch 00002: val_loss improved from 0.34005 to 0.27597, saving model to cnn_tf.model.best.hdf5
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.3649 - acc: 0.8660 - val_loss: 0.2760 - val_acc: 0.8972
     Epoch 3/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.3206 - acc: 0.8813  
     Epoch 00003: val_loss improved from 0.27597 to 0.25387, saving model to cnn_tf.model.best.hdf5
     55000/55000 [==============================] - 66s 1ms/step - loss: 0.3206 - acc: 0.8813 - val_loss: 0.2539 - val_acc: 0.9044
     Epoch 4/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2938 - acc: 0.8909  
     Epoch 00004: val_loss did not improve from 0.25387
     55000/55000 [==============================] - 66s 1ms/step - loss: 0.2938 - acc: 0.8909 - val_loss: 0.2543 - val_acc: 0.9054
     Epoch 5/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2750 - acc: 0.8992  
     Epoch 00005: val_loss improved from 0.25387 to 0.22453, saving model to cnn_tf.model.best.hdf5
     55000/55000 [==============================] - 66s 1ms/step - loss: 0.2750 - acc: 0.8992 - val_loss: 0.2245 - val_acc: 0.9158
     Epoch 6/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2633 - acc: 0.9017  
     Epoch 00006: val_loss did not improve from 0.22453
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.2635 - acc: 0.9017 - val_loss: 0.2312 - val_acc: 0.9106
     Epoch 7/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2554 - acc: 0.9047  
     Epoch 00007: val_loss improved from 0.22453 to 0.21361, saving model to cnn_tf.model.best.hdf5
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.2553 - acc: 0.9047 - val_loss: 0.2136 - val_acc: 0.9210
     Epoch 8/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2470 - acc: 0.9077  
     Epoch 00008: val_loss improved from 0.21361 to 0.20101, saving model to cnn_tf.model.best.hdf5
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.2471 - acc: 0.9077 - val_loss: 0.2010 - val_acc: 0.9234
     Epoch 9/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2374 - acc: 0.9108  
     Epoch 00009: val_loss did not improve from 0.20101
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.2374 - acc: 0.9108 - val_loss: 0.2104 - val_acc: 0.9196
     Epoch 10/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2295 - acc: 0.9141  
     Epoch 00010: val_loss did not improve from 0.20101
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.2295 - acc: 0.9141 - val_loss: 0.2030 - val_acc: 0.9222
     Epoch 11/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2297 - acc: 0.9145  
     Epoch 00011: val_loss did not improve from 0.20101
     55000/55000 [==============================] - 66s 1ms/step - loss: 0.2297 - acc: 0.9145 - val_loss: 0.2014 - val_acc: 0.9222
     Epoch 12/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2218 - acc: 0.9168  
     Epoch 00012: val_loss improved from 0.20101 to 0.19300, saving model to cnn_tf.model.best.hdf5
     55000/55000 [==============================] - 66s 1ms/step - loss: 0.2219 - acc: 0.9168 - val_loss: 0.1930 - val_acc: 0.9286
     Epoch 13/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2170 - acc: 0.9181  
     Epoch 00013: val_loss did not improve from 0.19300
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.2170 - acc: 0.9180 - val_loss: 0.2009 - val_acc: 0.9270
     Epoch 14/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2165 - acc: 0.9179  
     Epoch 00014: val_loss did not improve from 0.19300
     55000/55000 [==============================] - 66s 1ms/step - loss: 0.2165 - acc: 0.9179 - val_loss: 0.2058 - val_acc: 0.9246
     Epoch 15/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2123 - acc: 0.9195  
     Epoch 00015: val_loss did not improve from 0.19300
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.2123 - acc: 0.9195 - val_loss: 0.1971 - val_acc: 0.9240
     Epoch 16/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2088 - acc: 0.9204  
     Epoch 00016: val_loss improved from 0.19300 to 0.18862, saving model to cnn_tf.model.best.hdf5
     55000/55000 [==============================] - 68s 1ms/step - loss: 0.2089 - acc: 0.9204 - val_loss: 0.1886 - val_acc: 0.9344
     Epoch 17/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2058 - acc: 0.9233  
     Epoch 00017: val_loss did not improve from 0.18862
     55000/55000 [==============================] - 66s 1ms/step - loss: 0.2057 - acc: 0.9234 - val_loss: 0.1940 - val_acc: 0.9292
     Epoch 18/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2030 - acc: 0.9231  
     Epoch 00018: val_loss did not improve from 0.18862
     55000/55000 [==============================] - 67s 1ms/step - loss: 0.2030 - acc: 0.9231 - val_loss: 0.1951 - val_acc: 0.9304
     Epoch 19/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.1976 - acc: 0.9253  
     Epoch 00019: val_loss did not improve from 0.18862
     55000/55000 [==============================] - 65s 1ms/step - loss: 0.1975 - acc: 0.9254 - val_loss: 0.1948 - val_acc: 0.9306
     Epoch 20/20
     54976/55000 [============================>.] - ETA: 0s - loss: 0.2014 - acc: 0.9251  
     Epoch 00020: val_loss did not improve from 0.18862
     55000/55000 [==============================] - 65s 1ms/step - loss: 0.2014 - acc: 0.9251 - val_loss: 0.1933 - val_acc: 0.9266
     10000/10000 [==============================] - 2s 232us/step
     Test loss: 0.22195505007505417
     Test accuracy: 0.9202
     

# Prediction on test images  
<img src='https://alquarizm.files.wordpress.com/2019/01/screenshot-2019-01-14-at-2.42.36-am.png'/>
