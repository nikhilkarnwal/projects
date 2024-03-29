from skimage import io as img_io, color as img_color, transform, filters, restoration, util
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.color import rgb2gray
import skimage
import numpy as np
import os
import tensorflow_hub as hub


class DataSet:
    img_per_class = 1000
    category = []
    dim = (100, 100, 3)

    def __init__(self):
        self.category = []
        
    def load_dataset(self,
                     cat_file,
                     data_path,
                     dimension,
                     img_per_class=1000):
        self.dim = dimension
        print('Dimension of tensor image -', self.dim)
        self.img_per_class = img_per_class
        with open(cat_file, 'r') as cat_fp:
            for _cat in cat_fp:
                self.category.append(_cat.strip('\n'))
        print(self.category)
        self.load_images(data_path)

    def split_data(self, valid=0.2, test=0.2):
        x_test = []
        y_test = []
        x_valid = []
        y_valid = []
        x_train = []
        y_train = []

        n_total = len(self.x_train)
        n_test = int(n_total * test)
        n_valid = int((n_total - n_test) * valid)

        x_test.extend(self.x_train[:n_test])
        y_test.extend(self.y_train[:n_test])

        x_valid.extend(self.x_train[n_test:n_test + n_valid])
        y_valid.extend(self.y_train[n_test:n_test + n_valid])

        x_train.extend(self.x_train[n_test + n_valid:])
        y_train.extend(self.y_train[n_test + n_valid:])

        (self.x_train, self.y_train) = (np.asarray(x_train),
                                        np.asarray(y_train))
        (self.x_valid, self.y_valid) = (np.asarray(x_valid),
                                        np.asarray(y_valid))
        (self.x_test, self.y_test) = (np.asarray(x_test), np.asarray(y_test))
        self.dim = self.x_train[0].shape
        print(self.dim)
        print("Shape of train data %d by %d" % (len(x_train), len(y_train)))
        print("Shape of valid data %d by %d" % (len(x_valid), len(y_valid)))
        print("Shape of test data %d by %d" % (len(x_test), len(y_test)))

    def load_data(self):
        return (self.x_train, self.y_train), (self.x_valid,
                                              self.y_valid), (self.x_test,
                                                              self.y_test)

    def get_feature_vector(self, filename):
        return transform.resize(img_io.imread(filename),self.dim)

    def load_images(self, data_path):
        self.cat_to_id = {
            self.category[i]: i
            for i in range(len(self.category))
        }
        feat = []
        label = []
        x_train = []
        y_train = []
        for _cat in self.category:
            base_folder = data_path + '/' + _cat + '/'
            count = 0
            for filename in os.listdir(base_folder):
                #print(filename)
                if count >= self.img_per_class:
                    break
                count += 1
                img = self.get_feature_vector(base_folder + filename)
                x_train.append(img)
                y_train.append(self.cat_to_id[_cat])
                if count % 100 == 0:
                    print("Done with %d images" % count)
        for id in np.random.permutation(len(x_train)):
            feat.append(x_train[id])
            label.append(y_train[id])
        self.x_train = feat
        self.y_train = label
