from __future__ import absolute_import, division, print_function
from tensorflow.core.protobuf import config_pb2
import tensorflow as tf
import numpy as np
import os

LEARNING_RATE = 1e-4 / 2

class Unet_Model():
    def __init__(self, params):
        self.input_shape = [None, *params['image_shape'], 3]
        self.image_shape = params['image_shape']
        self.c_channels = [32, 64, 128, 256, 512]
        self.half_length = 4
        self.e_channels = self.c_channels[-2: :-1]

        self.c_layer = []
        self.c_size = []

    def __call__(self, inputs, training):
        conv = inputs

        # Contraction
        for b in range(4):
            for l in range(2):
                conv = tf.layers.conv2d(
                    inputs=conv, 
                    filters=self.c_channels[b],
                    kernel_size=[3, 3],
                    padding="SAME",
                    activation=tf.nn.relu)

            conv = tf.layers.batch_normalization(conv)
            self.c_layer.append(conv)  
            size = conv.get_shape().as_list()
            self.c_size.append(size)
            print('Encoder block {} last layer size {}'.format(
                b + 1, conv.get_shape().as_list()))
            conv = tf.layers.max_pooling2d(conv, [2, 2], [2, 2]) 
        
        conv = tf.layers.dropout(conv) # rate = 0.5 by default
        for l in range(2):
            conv = tf.layers.conv2d(
                inputs=conv, 
                filters=self.c_channels[self.half_length],
                kernel_size=[3, 3],
                padding="SAME",
                activation=tf.nn.relu)

        conv = tf.layers.batch_normalization(conv)
        print('Bottom block last layer size {}'.format(conv.get_shape().as_list()))

        # Expansion
        for b in range(4):
            conv = tf.layers.conv2d_transpose(
                inputs=conv,
                filters=self.e_channels[b],
                kernel_size=[2, 2],
                strides=[2, 2])
            size_current = conv.get_shape().as_list()
            skip = self.c_layer[-(b + 1)]
            size_old = self.c_size[-(b + 1)]                    
            conv = tf.concat([skip, conv], 3)

            for l in range(2):
                conv = tf.layers.conv2d(
                    inputs=conv,
                    filters=self.e_channels[b],
                    kernel_size=[3, 3],
                    padding="SAME",
                    activation=tf.nn.relu)

            conv = tf.layers.batch_normalization(conv)
            print('Decoder block {} last layer size {}'.format(
                b + 1, conv.get_shape().as_list()))

        logits = tf.layers.conv2d(
            inputs=conv,
            filters=2,
            kernel_size=[3, 3],
            padding="SAME")

        print('Logits layer size {}'.format(logits.get_shape().as_list()))
        return logits

def model_fn(features, labels, mode, params):
    PREDICT = tf.estimator.ModeKeys.PREDICT
    TRAIN   = tf.estimator.ModeKeys.TRAIN
    EVAL    = tf.estimator.ModeKeys.EVAL

    model = Unet_Model(params) # Instantiate the model
    images = features
    masks = labels

    if mode == PREDICT:
        logits = model(images, training=False) # Call the model
        predictions = {
            'image': images,
            'pred_mask': tf.argmax(logits, axis=-1),  # TODO: Verify
        }

        return tf.estimator.EstimatorSpec(
            mode=PREDICT,
            predictions=predictions)

    y_hot = tf.one_hot(masks, 2)

    if mode == TRAIN:
        logits = model(images, training=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y_hot)
        
        # Metrics are not yet supported during training when using Mirrored Strategy.
        # See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
        # accuracy = tf.metrics.accuracy(
        #     labels=masks, 
        #     predictions=tf.argmax(logits, axis=-1))

        return tf.estimator.EstimatorSpec(
            mode=TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    assert mode == EVAL
    logits = model(images, training=False)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y_hot)

    return tf.estimator.EstimatorSpec(
        mode=EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy': tf.metrics.accuracy(
                labels=masks, 
                predictions=tf.argmax(logits, axis=-1)) # TODO: Verify
        })
def parser(record, image_shape):
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
    
    segmented = tf.decode_raw(parsed["mask"], tf.uint8)
    # segmented = tf.cast(segmented, tf.float32)
    segmented = tf.reshape(segmented, shape=image_shape)

    return original, segmented

# TODO: Optimize the input_fn to reduce GPU idle time.
# This will probably require two separate input functions--one for
# training and the other for evaluation.
def input_fn(filenames, image_shape, train, num_repeat=1, batch_size=1, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(
        filenames=filenames, 
        compression_type="GZIP", # Full resolution images have been compressed.
        num_parallel_reads=8)

    if train:
        dataset.repeat(num_repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # dataset = dataset.map(lambda record: parser(record, image_shape))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        lambda record: parser(record, image_shape),
        batch_size=batch_size,
        num_parallel_batches=4))

    dataset = dataset.prefetch(buffer_size=None) # Let TF detect the optimal buffer_size
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md

    return dataset # Expected by estimator's `train` method.

def main():
    # TODO: Check if GPUs are available and make sure our model
    # is using the correct parameters. (Look at DS Comp code.)

    # Define parameters
    epochs = 100
    epochs_between_evals = 10 # When training on the full set of images, evaluation
                              # becomes a very expensive process. Use sparingly.

    # Define the location to save checkpoints:
    model_dir = os.path.expanduser('~/.temp/unet-big-full-resolution')

    # Specify where the training and evaluation files are located.
    train_tfrecord = \
        '/mnt/lfs2/pric7208/u-net/linh/BigData/Data/big_full_resolution_train.tfrecords'
    eval_tfrecord  = \
        '/mnt/lfs2/pric7208/u-net/linh/BigData/Data/big_full_resolution_eval.tfrecords'
    
    # Specify the shape of training and evaluation images.
    image_shape = [1280, 1920]

    # TODO: Finalize performance improvements.
    # Mirror the model accross all available GPUs using the mirrored distribution strategy.
    distribution = tf.contrib.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(
        train_distribute=distribution,
        keep_checkpoint_max=2,
        log_step_count_steps=5)

    # Initialize the Estimator
    image_segmentor = tf.estimator.Estimator(
        model_dir=model_dir,
        model_fn=model_fn,
        params={'image_shape': image_shape},
        config=config) # TODO: determine what parameters are needed

    # Train and evaluate
    for i in range(epochs // epochs_between_evals):
        print('\nEntering training epoch %d.\n' % (i * epochs_between_evals))
        image_segmentor.train(
            # input_fn is expected to take no arguments
            input_fn=lambda: input_fn(
                train_tfrecord,
                image_shape,
                train=True,
                num_repeat=epochs_between_evals))

        results = image_segmentor.evaluate(
            input_fn=lambda: input_fn(
                eval_tfrecord,
                image_shape,
                train=False))
        print('\nEvaluation results:\n%s\n' % results)

    print('Computing predictions...')
    predictions = image_segmentor.predict(
        input_fn=lambda: input_fn(eval_tfrecord, image_shape, train=False))
    image_list = []
    pred_mask_list = []
    for i, prediction in enumerate(predictions):
        if i % 10 == 0: print('  storing prediction %d' % i)
        image_list.append(prediction['image'])
        pred_mask_list.append(prediction['pred_mask'])
    np.save('big_eval_images.npy', np.asarray(image_list))
    np.save('big_eval_pred_masks.npy', np.asarray(pred_mask_list))

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
