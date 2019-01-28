This tutorial briefly talk about the implementation of Convolutional Neural Networks using tensorflow’s keras library. 
I have developed an Image Classifier using CNNs for classifying mnsit fashion dataset.


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
