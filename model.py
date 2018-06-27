import tensorflow as tf

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
        activation=tf.nn.relu,
        name=name)
        # TODO: Do we need to specify ReLU activation here?
        # Let's find out...
    return layer

class Unet_Model():
    def __init__(self, params):
        self.data_format = params['data_format']

        self.num_conv_filters_at_level = [32, 64, 128, 256, 512]
        # self.num_conv_filters_at_level = [20, 30, 40, 50, 80]
        self.num_max_pool_ops = 4
        self.num_up_conv_ops = self.num_max_pool_ops
        self.num_conv_ops_per_level = 2

        # TODO: Finish renaming
        self.e_channels = self.num_conv_filters_at_level[-2: :-1]

        self.c_layer = []
        self.c_size = []

    def __call__(self, inputs, training):
        # Note: We currently don't use the `training` argument.
        # This argument could be used in the future if we wish to add Dropout.
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
    num_output_classes = params['num_output_classes']
    learning_rate = params['learning_rate']

    channels_axis = (1 if data_format == 'channels_first' else -1)

    # TODO: Rewrite the prediction logic to work with the new data_format.
    if mode == PREDICT:
        logits = model(images, training=False) # Call the model

        class_probabilities = tf.nn.softmax(logits, axis=channels_axis)
        
        # Convert probabilities into a color image to examine network confidence.
        # Here we decide to create a blue-green image; the red channel is set to zero.
        if data_format == 'channels_first':
            b_channels = class_probabilities[:, 0, :, :]
            g_channels = class_probabilities[:, 1, :, :]
        else:
            b_channels = class_probabilities[:, :, :, 0]
            g_channels = class_probabilities[:, :, :, 1]

        r_channels = tf.zeros(b_channels.shape)
        rgb_images = tf.stack([r_channels, g_channels, b_channels], axis=channels_axis)
        rgb_images = tf.cast(rgb_images * 255, tf.uint8)

        predicted_masks = tf.argmax(logits, axis=channels_axis)
        
        pixel_matches = tf.equal(predicted_masks, masks)
        pixel_matches = tf.cast(pixel_matches, tf.float32)
        accuracy = tf.reduce_mean(pixel_matches)
        # Note: This computes accuracy over the whole batch.
        
        predictions = {
            'images': images,
            'heat_maps': rgb_image,
            'masks': predicted_masks,
            'accuracy': accuracy
        }

        return tf.estimator.EstimatorSpec(
            mode=PREDICT,
            predictions=predictions)

    onehot_masks = tf.one_hot(
        masks,folds = KFolds(IMAGE_FILENAMES, MASK_FILENAMES,
        num_folds=NUM_FOLDS, sort=False, yield_dict=False)

    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    params = {'data_format': data_format}

    image_segmentor = tf.estimator.Estimator(
        model_dir='-'.join([MODEL_DIR, str(FOLD_NUM)]),
        model_fn=model_fn,
        params=params)

    # Fetch the images in the evaluation set of the specified fold.
    # Note: We throw away the training images and training masks.
    (_, _), (eval_images, eval_masks) = folds.get_fold(FOLD_NUM)

    # Select a few of the evaluation images at random.
    num_images = len(eval_images)
    assert num_images == len(eval_masks)

    random_indexes = np.random.choice(num_images, size=NUM_IMAGES_TO_PREDICT, replace=False)
    images_to_predict = eval_images[random_indexes]
    images_masks = eval_masks[random_indexes]
        depth=num_output_classes,
        axis=(1 if data_format == 'channels_first' else -1))

    if mode == TRAIN:
        logits = model(images, training=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

