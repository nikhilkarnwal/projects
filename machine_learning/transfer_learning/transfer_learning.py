from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class TransferLearning:
    FLAGS = None

    MAX_NUM_IMAGES_PER_CLASS = 2**27 - 1  # ~134M

    # The location where variable checkpoints will be stored.
    CHECKPOINT_NAME = './tmp/_retrain_checkpoint'

    # A module is understood as instrumented for quantization with TF-Lite
    # if it contains any of these ops.
    FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                      'FakeQuantWithMinMaxVarsPerChannel')

    def load_model(self):
        height, width = hub.get_expected_image_size(self.model_spec)
        with tf.Graph().as_default() as graph:
            resized_input_tensor = tf.placeholder(tf.float32,
                                                  [None, height, width, 3])
            m = hub.Module(self.model_spec)
            bottleneck_tensor = m(resized_input_tensor)
            wants_quantization = any(node.op in self.FAKE_QUANT_OPS
                                     for node in graph.as_graph_def().node)
        self.graph = graph
        self.bottleneck_tensor = bottleneck_tensor
        self.input_tensor = resized_input_tensor
        self.wants_quantization = wants_quantization

    def run_bottleneck_on_image(self, image_data):
        resized_input_values = self.sess.run(
            self.decoded_image_tensor, {self.image_data_tensor: image_data})
        bottleneck_values = self.sess.run(
            self.bottleneck_tensor, {self.input_tensor: resized_input_values})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    def image_to_feature(self, image_path):
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = self.run_bottleneck_on_image(image_data)
        return bottleneck_values

    def main(
            self,
            module_name='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
    ):
        print('Loading module %s' %module_name)
        self.model_spec = hub.load_module_spec(module_name)
        print('Loading model')
        self.load_model()
        print('Model Loaded',self.graph)
        with self.graph.as_default():
            init = tf.global_variables_initializer()
        #self.graph.finalize()
        sess = tf.Session(graph = self.graph)
        
        # Initialize all weights: for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values.
        with sess.as_default():
            pass
        sess.run(init)
        self.sess = sess
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = self.add_jpeg_decoding()
        print('Intialized graph')
        print(sess.graph)
        self.image_data_tensor = jpeg_data_tensor
        self.decoded_image_tensor = decoded_image_tensor

    def close(self):
        if not self.sess._closed:
            self.sess.close()
            print('Closing session')
        else:
            print('Session already closed')

    def add_jpeg_decoding(self):
        input_height, input_width = hub.get_expected_image_size(
            self.model_spec)
        input_depth = hub.get_num_image_channels(self.model_spec)
        with self.graph.as_default():
            jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
            decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
            # Convert from full range of uint8 to range [0,1] of float32.
            decoded_image_as_float = tf.image.convert_image_dtype(
                decoded_image, tf.float32)
            decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
            resize_shape = tf.stack([input_height, input_width])
            resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
            resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                                 resize_shape_as_int)
        return jpeg_data, resized_image
