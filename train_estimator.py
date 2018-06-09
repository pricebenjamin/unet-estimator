from __future__ import absolute_import, division, print_function
from tensorflow.core.protobuf import config_pb2
from PIL import Image
from multiprocessing import Pool

import tensorflow as tf
import numpy as np
import sys
import os

LEARNING_RATE = 1e-5

# Define parameters
epochs = 20 
epochs_between_evals = 2 # When training on the full set of images, evaluation
                         # becomes a very expensive process. Use sparingly.

# Define the location to save checkpoints:
# model_dir = os.path.expanduser('~/.temp/unet-big-full-resolution')
model_dir = os.path.expanduser('~/.temp/unet-full-resolution-latest')

# Specify where the training and evaluation files are located.
train_tfrecord = \
    '/mnt/lfs2/pric7208/u-net/linh/BigData/Data/big_full_resolution_train.tfrecords'
    # '/mnt/lfs2/pric7208/repos/unet-estimator/small_full_resolution_train.tfrecords'
eval_tfrecord  = \
    '/mnt/lfs2/pric7208/u-net/linh/BigData/Data/big_full_resolution_eval.tfrecords'
    # '/mnt/lfs2/pric7208/repos/unet-estimator/small_full_resolution_eval.tfrecords'

# Specify the shape of training and evaluation images.
image_shape = [1280, 1920]
num_channels = 3 # Number of channels in the input image

# TODO: Finalize performance improvements.
# Mirror the model accross all available GPUs using the mirrored distribution strategy.
distribution = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
    train_distribute=distribution,
    keep_checkpoint_max=2,
    log_step_count_steps=5)

# Create a simple wrapper for constructing convolutional layers with particular settings.
# Our network frequently uses [3,3] kernels, [1,1] strides, and ReLU activations.
def conv_layer(num_filters, data_format, name):
    layer = tf.layers.Conv2D(
        filters=num_filters,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name=name)
    return layer

# Create a similar wrapper for the "upsampling" layers.
def conv_transpose_layer(num_filters, data_format, name):
    layer = tf.layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=[2, 2],
        strides=[2, 2],
        data_format=data_format,
        name=name)
        # TODO: Do we need to specify ReLU activation here?
    return layer

class Unet_Model():
    def __init__(self, params):
        self.data_format = params['data_format']
        num_channels = params['num_channels']
        image_height, image_width = params['image_shape']

        if self.data_format == 'channels_first':
            self.input_shape = [None, num_channels, image_height, image_width]
        else:
            assert self.data_format == 'channels_last'
            self.input_shape = [None, image_height, image_width, num_channels]

        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = num_channels

        self.num_conv_filters_at_level = [32, 64, 128, 256, 512]
        self.num_max_pool_ops = 4
        self.num_up_conv_ops = self.num_max_pool_ops
        self.num_conv_ops_per_level = 2
        self.e_channels = self.num_conv_filters_at_level[-2: :-1]

        self.c_layer = []
        self.c_size = []

    def __call__(self, inputs, training):
        conv = inputs

        # Contraction
        for max_pool_op in range(self.num_max_pool_ops):
            for conv_op in range(self.num_conv_ops_per_level):
               # Instantiate the layer with desired parameters.
               layer = conv_layer(
                   num_filters=self.num_conv_filters_at_level[max_pool_op],
                   data_format=self.data_format,
                   name='Conv2D_C_{}{}'.format(max_pool_op, conv_op))

               # Apply the layer to the inputs.
               conv = layer(inputs=conv)

            # Store the last layer of each level.
            self.c_layer.append(conv)
            size = conv.get_shape().as_list()
            self.c_size.append(size)

            print('Encoder block {} last layer size {}'.format(
                max_pool_op + 1, size))

            # Apply max pooling at the end of each level.
            conv = tf.layers.max_pooling2d(
                conv, [2, 2], [2, 2], 
                data_format=self.data_format,
                name='MaxPool_{}'.format(max_pool_op))
        
        # Bottom convolutional layer
        for conv_op in range(self.num_conv_ops_per_level):
            layer = conv_layer(
                num_filters=self.num_conv_filters_at_level[-1],
                data_format=self.data_format,
                name='Conv2D_Bottom_{}'.format(conv_op))

            conv = layer(inputs=conv)

        print('Bottom block last layer size {}'.format(conv.get_shape().as_list()))

        # Expansion
        for up_conv_op in range(self.num_up_conv_ops):
            # Instantiate the layer with desired parameters.
            layer = conv_transpose_layer(
                num_filters=self.e_channels[up_conv_op],
                data_format=self.data_format,
                name='Upsample_{}'.format(up_conv_op))

            # Apply the layer to the input.
            conv = layer(inputs=conv)

            # Concatenate with contraction outputs.
            size_current = conv.get_shape().as_list()
            skip = self.c_layer[-(up_conv_op + 1)]
            size_old = self.c_size[-(up_conv_op + 1)]                    

            conv = tf.concat([skip, conv],
                axis=(1 if self.data_format == 'channels_first' else -1))

            for conv_op in range(self.num_conv_ops_per_level):
                # Instantiate the convolutional layer.
                layer = conv_layer(
                    num_filters=self.e_channels[up_conv_op],
                    data_format=self.data_format,
                    name='Conv2D_E_{}{}'.format(up_conv_op, conv_op))

                # Apply the layer to the inputs
                conv = layer(inputs=conv)

            print('Decoder block {} last layer size {}'.format(
                up_conv_op + 1, conv.get_shape().as_list()))

        logits = tf.layers.conv2d(
            inputs=conv,
            filters=2,
            kernel_size=[3, 3],
            padding="SAME",
            data_format=self.data_format)

        print('Logits layer size {}'.format(logits.get_shape().as_list()))
        return logits

def model_fn(features, labels, mode, params):
    PREDICT = tf.estimator.ModeKeys.PREDICT
    TRAIN   = tf.estimator.ModeKeys.TRAIN
    EVAL    = tf.estimator.ModeKeys.EVAL

    model = Unet_Model(params) # Instantiate the model
    images = features
    masks = labels
    data_format = params['data_format']

    # TODO: Rewrite the prediction logic to work with the new data_format.
    if mode == PREDICT:
        logits = model(images, training=False) # Call the model
        pred_masks = tf.argmax(logits, axis=-1)
        pred_masks = tf.stack([pred_masks] * 3, axis=-1)
        pred_masks = tf.cast(pred_masks, tf.float32)
        predictions = {
            # 'image': images,
            # 'pred_mask': tf.argmax(logits, axis=-1),
            'cut': tf.multiply(images, pred_masks)
        }

        return tf.estimator.EstimatorSpec(
            mode=PREDICT,
            predictions=predictions)

    y_hot = tf.one_hot(masks, 2,
                       axis=(1 if data_format == 'channels_first' else -1))

    if mode == TRAIN:
        logits = model(images, training=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_hot,
                logits=logits,
                dim=(1 if data_format == 'channels_first' else -1)))

        # Metrics are not yet supported during training when using Mirrored Strategy.
        # See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
        # accuracy = tf.metrics.accuracy(
        #     labels=masks, 
        #     predictions=tf.argmax(logits, axis=-1))

        return tf.estimator.EstimatorSpec(
            mode=TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    # TODO: consider wrapping the loss computation in a function to avoid rewriting it.
    assert mode == EVAL
    logits = model(images, training=False)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_hot,
            logits=logits,
            dim=(1 if data_format == 'channels_first' else -1)))

    return tf.estimator.EstimatorSpec(
        mode=EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy': tf.metrics.accuracy(
                labels=masks, 
                predictions=tf.argmax(logits, axis=(1 if data_format == 'channels_first' else -1)))
        })

def parser(record, image_shape, data_format):
    '''
    Defines how to convert each record in a TFRecords dataset into
    its original form.

    For our network, each record contains an original image with shape
    (n, n, 3) and a segmented image (mask) of shape (n, n).

    Returns a tuple: (image, mask)
    '''

    keys_to_features = {
        "image":  tf.FixedLenFeature([], tf.string),
        "mask": tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    original = tf.decode_raw(parsed["image"], tf.uint8)
    original = tf.cast(original, tf.float32)

    original = tf.reshape(original, shape=[*image_shape, 3])

    if data_format == 'channels_first':
        # Channels first should improve performance when training on GPU
        # original = tf.reshape(original, shape=[3, *image_shape])
        red   = original[:, :, 0]
        green = original[:, :, 1]
        blue  = original[:, :, 2]
        original = tf.stack([red, green, blue], axis=0)
        # TODO: Experiment with reshaping to determine if this is valid.
        # These reshape commands may be causing problems for training. The images were
        # saved in HWC format; reshaping them may destroy the image.
   
    segmented = tf.decode_raw(parsed["mask"], tf.uint8)
    segmented = tf.cast(segmented, tf.int32)
    segmented = tf.reshape(segmented, shape=image_shape)
    return original, segmented

def input_fn(filename, image_shape, data_format, train, num_repeat=1, batch_size=1):
    # Training Performance: A user's guid to converge faster (TF Dev Summit 2018)
    # https://www.youtube.com/watch?v=SxOsJPaxHME&t=1529s
    dataset = tf.data.TFRecordDataset(
        filenames=filename, 
        compression_type="GZIP", # Full resolution images have been compressed.
        num_parallel_reads=8)

    if train: # TODO: Create and examine a profile trace
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, num_repeat))
        # Does this actually load the data into memory? Should we not store our data
        # in a TFRecord file? We could just point TF to the data directory.
    else:
        dataset = dataset.repeat(num_repeat)

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        lambda record: parser(record, image_shape, data_format),
        batch_size=batch_size,
        num_parallel_batches=8))

    dataset = dataset.prefetch(4)
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    return dataset # Expected by estimator's `train` method.

def main():
    # Check if the system's version of TensorFlow was built with CUDA (i.e. uses a GPU)
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    params = {
        'image_shape': image_shape, # Defined after import statements
        'num_channels': num_channels,
        'data_format': data_format,
    }

    # Initialize the Estimator
    image_segmentor = tf.estimator.Estimator(
        model_dir=model_dir,
        model_fn=model_fn,
        params=params,
        config=config)

    # Train and evaluate
    for i in range(epochs // epochs_between_evals):
        print('\nEntering training epoch %d.\n' % (i * epochs_between_evals))
        image_segmentor.train(
            # input_fn is expected to take no arguments
            input_fn=lambda: input_fn(
                train_tfrecord,
                image_shape,
                data_format,
                train=True,
                num_repeat=epochs_between_evals))

        results = image_segmentor.evaluate(
            input_fn=lambda: input_fn(
                eval_tfrecord,
                image_shape,
                data_format,
                train=False))
        print('\nEvaluation results:\n%s\n' % results)

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
