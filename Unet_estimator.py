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
        self.e_channels = self.c_channels[-2: : -1]

        self.c_layer = []
        self.c_size = []

    def __call__(self, inputs, training):
        conv = inputs # tf.placeholder(tf.float32, [None, 512, 512, 3])

        # Contraction
        for b in range(4):
            for l in range(2):
                conv = tf.layers.conv2d(
                    inputs=conv, 
                    filters=self.c_channels[b],
                    kernel_size=[3, 3],
                    padding="SAME",
                    activation=tf.nn.relu)

            # conv = tf.layers.batch_normalization(conv)
            self.c_layer.append(conv)  
            size = conv.get_shape().as_list()
            self.c_size.append(size)
            print('Encoder block {} last layer size {}'.format(b+1,conv.get_shape().as_list()))
            conv = tf.layers.max_pooling2d(conv,[2,2],[2,2]) 
        
        conv = tf.layers.dropout(conv) # rate = 0.5 by default
        for l in range(2):
            conv = tf.layers.conv2d(
                inputs=conv, 
                filters=self.c_channels[self.half_length],
                kernel_size=[3, 3],
                padding="SAME",
                activation=tf.nn.relu)

        # conv = tf.layers.batch_normalization(conv)
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

            # conv = tf.layers.batch_normalization(conv)
            print('Decoder block {} last layer size {}'.format(b + 1, conv.get_shape().as_list()))

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

    if mode == TRAIN:
        logits = model(images, training=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) # TODO: Define
        #loss = tf.losses.softmax_cross_entropy(labels=masks, logits=logits)
        hot = tf.one_hot(masks, 2)
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=hot)
        accuracy = tf.metrics.accuracy(
            labels=masks, 
            predictions=tf.argmax(logits, axis=-1))

        return tf.estimator.EstimatorSpec(
            mode=TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    assert mode == EVAL
    logits = model(images, training=False)
    hot = tf.one_hot(masks, 2)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=hot)

    return tf.estimator.EstimatorSpec(
        mode=EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy': tf.metrics.accuracy(
                labels=masks, 
                predictions=tf.argmax(logits, axis=-1)) # TODO: Verify
        })

def parser(record, image_shape): # TODO: determine how/where to define `n`; partially solved
    '''
    Defines how to convert each record in a TFRecords dataset into
    its original form.

    For our network, each record contains an original image with shape
    (n, n, 3) and a segmented image (mask) of shape (n, n).

    Returns a tuple: (image, mask)
    '''

    keys_to_features = {
        "original":  tf.FixedLenFeature([], tf.string),
        "segmented": tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    original = tf.decode_raw(parsed["original"], tf.uint8)
    original = tf.cast(original, tf.float32)
    original = tf.reshape(original, shape=[*image_shape, 3])
    
    segmented = tf.decode_raw(parsed["segmented"], tf.uint8)
    # segmented = tf.cast(segmented, tf.float32)
    segmented = tf.reshape(segmented, shape=image_shape)

    return original, segmented

# TODO: Optimize the input_fn to reduce GPU idle time.
# This will probably require two separate input functions--one for
# training and the other for evaluation.
def input_fn(filenames, image_shape, train, batch_size=2, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(lambda record: parser(record, image_shape))

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = 3
    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    return dataset # Expected by estimator's `train` method.


def main():
    # TODO: Check if GPUs are available and make sure our model
    # is using the correct parameters. (Look at DS Comp code.)

    # Define parameters
    epochs = 100 
    epochs_between_evals = 2
    model_dir = os.path.expanduser('~/.temp/unet')
    train_tfrecord = 'train_512.tfrecords'
    eval_tfrecord  = 'val_512.tfrecords'
    image_shape = [512, 512]

    if not os.path.exists(model_dir):
        print('Model directory `%s` does not exist.' % model_dir)
        print('You may need to create this directory before running this program.')

    # TODO: Finalize performance improvements.
    # distribution = tf.contrib.distribute.MirroredStrategy() # Mirrors the model
    # accross all available GPUs.

    config = tf.estimator.RunConfig(
        # train_distribute=distribution,
        keep_checkpoint_max=2,
        log_step_count_steps=10
        )

    # Initialize the Estimator
    image_segmentor = tf.estimator.Estimator(
        model_dir=model_dir,
        model_fn=model_fn, # TODO: Finalize
        params={'image_shape': image_shape},
        config=config) # TODO: determine what parameters are needed

    # Train and evaluate
    for i in range(epochs // epochs_between_evals):
        print('\nEntering training epoch %d.\n' % (i * epochs_between_evals))
        image_segmentor.train(
            input_fn=lambda: input_fn(train_tfrecord, image_shape, train=True)) # TODO

        results = image_segmentor.evaluate(
            input_fn=lambda: input_fn(eval_tfrecord, image_shape, train=False)) # TODO
        print('\nEvaluation results:\n%s\n' % results)

    print('Computing predictions...')
    predictions = image_segmentor.predict(
        input_fn=lambda: input_fn(eval_tfrecord, image_shape, train=False)) # TODO
    image_list = []
    pred_mask_list = []
    for i, prediction in enumerate(predictions):
        if i % 10 == 0: print('  storing prediction %d' % i)
        image_list.append(prediction['image'])
        pred_mask_list.append(prediction['pred_mask'])
    np.save('eval_images.npy', np.asarray(image_list))
    np.save('eval_pred_masks.npy', np.asarray(pred_mask_list))

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
