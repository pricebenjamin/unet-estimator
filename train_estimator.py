from __future__ import absolute_import, division, print_function
from tensorflow.core.protobuf import config_pb2
from PIL import Image
from multiprocessing import Pool
from glob import glob

import tensorflow as tf
import numpy as np
import sys
import os

LEARNING_RATE = 1e-5

MODEL_DIR = '/mnt/lfs2/pric7208/tf-saves/shallow-unet-estimator'
NUM_EPOCHS = 20
EPOCHS_BETWEEN_EVALS = 2

WORKING_DIR = '/mnt/lfs2/pric7208/kaggle/carvana'
IMAGE_DIR = os.path.join(WORKING_DIR, 'train_hq')
MASK_DIR  = os.path.join(WORKING_DIR, 'train_masks')

IMAGE_FILENAMES = sorted(glob(os.path.join(IMAGE_DIR, '*.jpg')))
MASK_FILENAMES = sorted(glob(os.path.join(MASK_DIR, '*.gif')))

INITIAL_IMAGE_SHAPE = [1280, 1918] # Used by `parser` to pad the image
TARGET_IMAGE_SHAPE  = [1280, 1920] # to a usable size (divisible by 32)
NUM_CHANNELS = 3

NUM_OUTPUT_CLASSES = 2 # Output pixels are either 'on' or 'off'

NUM_PARALLEL_BATCHES = 8
NUM_PREFETCH = None

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

        # self.num_conv_filters_at_level = [32, 64, 128, 256, 512]
        self.num_conv_filters_at_level = [16, 32, 64, 128]
        # self.num_max_pool_ops = 4
        self.num_max_pool_ops = 3
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

def compute_loss(logits, onehot_masks, data_format):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=onehot_masks,
            logits=logits,
            dim=(1 if data_format == 'channels_first' else -1))) # Specify class dimension

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

    onehot_masks = tf.one_hot(
        masks,
        depth=NUM_OUTPUT_CLASSES,
        axis=(1 if data_format == 'channels_first' else -1))

    if mode == TRAIN:
        logits = model(images, training=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        loss = compute_loss(logits, onehot_masks, data_format)

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
    loss = compute_loss(logits, onehot_masks, data_format)

    return tf.estimator.EstimatorSpec(
        mode=EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy': tf.metrics.accuracy(
                labels=masks, 
                predictions=tf.argmax(logits, axis=(1 if data_format == 'channels_first' else -1)))
        })

# def parser(record, image_shape, data_format):
#     '''
#     Defines how to convert each record in a TFRecords dataset into
#     its original form.
# 
#     For our network, each record contains an original image with shape
#     (n, n, 3) and a segmented image (mask) of shape (n, n).
# 
#     Returns a tuple: (image, mask)
#     '''
# 
#     keys_to_features = {
#         "image":  tf.FixedLenFeature([], tf.string),
#         "mask": tf.FixedLenFeature([], tf.string)
#     }
#     parsed = tf.parse_single_example(record, keys_to_features)
#     original = tf.decode_raw(parsed["image"], tf.uint8)
#     original = tf.cast(original, tf.float32)
# 
#     original = tf.reshape(original, shape=[*image_shape, 3])
# 
#     if data_format == 'channels_first':
#         # Channels first should improve performance when training on GPU
#         # original = tf.reshape(original, shape=[3, *image_shape])
#         red   = original[:, :, 0]
#         green = original[:, :, 1]
#         blue  = original[:, :, 2]
#         original = tf.stack([red, green, blue], axis=0)
#         # TODO: Experiment with reshaping to determine if this is valid.
#         # These reshape commands may be causing problems for training. The images were
#         # saved in HWC format; reshaping them may destroy the image.
#    
#     segmented = tf.decode_raw(parsed["mask"], tf.uint8)
#     segmented = tf.cast(segmented, tf.int32)
#     segmented = tf.reshape(segmented, shape=image_shape)
#     return original, segmented

def parser(single_image_filename_tensor, single_mask_filename_tensor, data_format):
    # Parse the original image into a tensor of shape == [1280, 1920, 3]
    # with dtype == tf.float32 and RGB values in the range [0, 1].
    image = tf.read_file(single_image_filename_tensor)
    image = tf.image.decode_jpeg(image, channels=NUM_CHANNELS)
    # After decoding, image.shape == [1280, 1918, 3]
    top_padding  = int((TARGET_IMAGE_SHAPE[0] - INITIAL_IMAGE_SHAPE[0]) / 2)
    left_padding = int((TARGET_IMAGE_SHAPE[1] - INITIAL_IMAGE_SHAPE[1]) / 2)
    image = tf.image.pad_to_bounding_box(
        # Pads the image to the desired size by adding zeros.
        image,
        # Adds `offset_height` rows of zeros to the top
        offset_height=top_padding,
        # Adds `offset_width` rows of zeros to the left
        offset_width=left_padding,
        # Pad the image on the bottom until `target_height`
        target_height=TARGET_IMAGE_SHAPE[0],
        # Pad the iamge on the right until `target_width`
        target_width=TARGET_IMAGE_SHAPE[1])

    if data_format == 'channels_first':
        c0 = image[:, :, 0]
        c1 = image[:, :, 1]
        c2 = image[:, :, 2]
        image = tf.stack([c0, c1, c2], axis=0)

    image = tf.cast(image, tf.float32) / 255 # Convert and scale

    # Parse the segmented image (mask) into a tensor of shape == [1280, 1920]
    # with dtype == tf.int32 and grayscale values in the rage [0, 1].
    mask = tf.read_file(single_mask_filename_tensor)
    mask = tf.image.decode_gif(mask)
    # After decoding, mask.shape == [1, 1280, 1918, 3]
    mask = tf.image.rgb_to_grayscale(mask)
    # After grayscaling, mask.shape == [1, 1280, 1918, 1]
    mask = tf.image.pad_to_bounding_box( # Works with 4-D tensors
        mask,
        offset_height=top_padding,
        offset_width=left_padding,
        target_height=TARGET_IMAGE_SHAPE[0],
        target_width=TARGET_IMAGE_SHAPE[1])
    mask = tf.reshape(mask, TARGET_IMAGE_SHAPE)
    # After reshaping, mask.shape == [1280, 1920]
    mask = tf.cast(mask / 255, tf.int32)
    return image, mask

# def input_fn(filename, image_shape, data_format, train, num_repeat=1, batch_size=1):
#     # Training Performance: A user's guid to converge faster (TF Dev Summit 2018)
#     # https://www.youtube.com/watch?v=SxOsJPaxHME&t=1529s
#     dataset = tf.data.TFRecordDataset(
#         filenames=filename, 
#         compression_type="GZIP", # Full resolution images have been compressed.
#         num_parallel_reads=8)
# 
#     if train: # TODO: Create and examine a profile trace
#         dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, num_repeat))
#         # Does this actually load the data into memory? Should we not store our data
#         # in a TFRecord file? We could just point TF to the data directory.
#     else:
#         dataset = dataset.repeat(num_repeat)
# 
#     dataset = dataset.apply(tf.contrib.data.map_and_batch(
#         lambda record: parser(record, image_shape, data_format),
#         batch_size=batch_size,
#         num_parallel_batches=8))
# 
#     dataset = dataset.prefetch(4)
#     # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
#     return dataset # Expected by estimator's `train` method.

def input_fn(training, data_format, num_repeats=1, batch_size=1):
    image_filename_tensor = tf.data.Dataset.from_tensor_slices(IMAGE_FILENAMES)
    mask_filename_tensor  = tf.data.Dataset.from_tensor_slices(MASK_FILENAMES)
    examples = tf.data.Dataset.zip(
        (image_filename_tensor, mask_filename_tensor))

    if training:
        examples = examples.apply(tf.contrib.data.shuffle_and_repeat(
            len(IMAGE_FILENAMES) * num_repeats,
            num_repeats))
    
    examples = examples.apply(tf.contrib.data.map_and_batch(
        lambda image, mask: parser(image, mask, data_format),
        batch_size=batch_size,
        num_parallel_batches=NUM_PARALLEL_BATCHES))
    
    examples = examples.prefetch(NUM_PREFETCH)
    return examples

def main():
    # Check if the system's version of TensorFlow was built with CUDA (i.e. uses a GPU)
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    params = {
        'data_format': data_format,
    }

    # Initialize the Estimator
    image_segmentor = tf.estimator.Estimator(
        model_dir=MODEL_DIR,
        model_fn=model_fn,
        params=params,
        config=config)

    # Train and evaluate
    for i in range(NUM_EPOCHS // EPOCHS_BETWEEN_EVALS):
        print('\nEntering training epoch %d.\n' % (i * EPOCHS_BETWEEN_EVALS))
        image_segmentor.train(
            # input_fn is expected to take no arguments
            input_fn=lambda: input_fn(
                training=True,
                data_format=params['data_format'],
                num_repeats=EPOCHS_BETWEEN_EVALS,
                batch_size=1))

        results = image_segmentor.evaluate(
            input_fn=lambda: input_fn(
                training=False,
                data_format=params['data_format']))

        print('\nEvaluation results:\n%s\n' % results)

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
