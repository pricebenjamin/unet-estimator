import tensorflow as tf

def parser(
    single_image_filename_tensor, # A tensor containing a single filename
    single_mask_filename_tensor,  # A tensor containing a single filename
    data_format,         # If 'channels_first', reshape the image
    pad_image,           # Boolean: Should we pad the image?
    initial_image_shape, 
    target_image_shape,  # Pad with zeros until target shape
    num_channels=3):     # Number of channels in the input image
    '''
    TODO: docstring

    Returns: (`image`, `mask`) if single_mask_filename_tensor is not None, other-
    wise returns `image`.
    '''

    # TODO: Consider examining performance if TF operations are replaced with
    # pillow (or opencv) operations, such as Image.open or Image.resize
    
    # Parse the original image into a tensor of shape == [1280, 1920, 3]
    # with dtype == tf.float32 and RGB values in the range [0, 1].
    image = tf.read_file(single_image_filename_tensor)
    image = tf.image.decode_jpeg(image, channels=num_channels)
    # After decoding, image.shape == [1280, 1918, 3]

    top_padding  = int((target_image_shape[0] - initial_image_shape[0]) / 2)
    left_padding = int((target_image_shape[1] - initial_image_shape[1]) / 2)

    if pad_image:
        image = tf.image.pad_to_bounding_box(
            # Pads the image to the desired size by adding zeros.
            image,
            # Adds `offset_height` rows of zeros to the top
            offset_height=top_padding,
            # Adds `offset_width` rows of zeros to the left
            offset_width=left_padding,
            # Pad the image on the bottom until `target_height`
            target_height=target_image_shape[0],
            # Pad the iamge on the right until `target_width`
            target_width=target_image_shape[1])

    if data_format == 'channels_first':
        # Extract each channel in an explicit way to prevent
        # mangling the structure of the image.
        c0 = image[:, :, 0]
        c1 = image[:, :, 1]
        c2 = image[:, :, 2]
        # Recombine the channels
        image = tf.stack([c0, c1, c2], axis=0)

    image = tf.cast(image, tf.float32) / 255 # Convert and scale
    # TODO: Does scaling provide any benefit? Can we test this?

    if single_mask_filename_tensor is not None:
        # Parse the segmented image (mask) into a tensor of shape == [1280, 1920]
        # with dtype == tf.int32 and grayscale values in the rage [0, 1].
        mask = tf.read_file(single_mask_filename_tensor)
        mask = tf.image.decode_gif(mask)
        # After decoding, mask.shape == [1, 1280, 1918, 3]
        # This is because 'gif' files are assumed to contain several 'frames'.
        mask = tf.image.rgb_to_grayscale(mask)
        # After grayscaling, mask.shape == [1, 1280, 1918, 1]

        if pad_image:
            mask = tf.image.pad_to_bounding_box( # Works with 4-D tensors
                mask,
                offset_height=top_padding,
                offset_width=left_padding,
                target_height=target_image_shape[0],
                target_width=target_image_shape[1])
            mask = tf.reshape(mask, target_image_shape)
            # After reshaping, mask.shape == [1280, 1920]
        else:
            shape = tf.shape(mask)[1:-1] # Remove extraneous dimensions
            mask = tf.reshape(mask, shape)

        mask = tf.cast(mask / 255, tf.int32)
        return image, mask
    else:
        return image

def input_fn(
    image_filenames,  # List of image filenames
    mask_filenames,   # List of mask filenames
    training,         # Determines whether or not images should be shuffled
    data_format,      # Passed to the parser so that RGB images can be reshaped
    pad_image=True,   # Passed to the parser
    initial_image_shape=[1280, 1918], # Passed to the parser
    target_image_shape=[1280, 1920],  # Passed to the parser
    num_repeats=1,    # Number of times to repeat the set of images
    batch_size=1,     # Number of images in each batch
    num_parallel_batches=8, # How many processing cores are available?
    num_prefetch=None):     # Number of images to pretech (None: let TF decide)
    
    image_filename_tensor = tf.data.Dataset.from_tensor_slices(image_filenames)
    mask_filename_tensor  = tf.data.Dataset.from_tensor_slices(mask_filenames)
    examples = tf.data.Dataset.zip(
        (image_filename_tensor, mask_filename_tensor))

    if training:
        examples = examples.apply(tf.contrib.data.shuffle_and_repeat(
            len(image_filenames) * num_repeats, # buffer size
            num_repeats))
    
    examples = examples.apply(tf.contrib.data.map_and_batch(
        lambda image, mask: parser(image, mask, 
            data_format=data_format,
            pad_image=pad_image,
            initial_image_shape=initial_image_shape,
            target_image_shape=target_image_shape),
        batch_size=batch_size,
        num_parallel_batches=num_parallel_batches))
    
    examples = examples.prefetch(num_prefetch)
    return examples

def prediction_input_fn(
    image_filenames,
    data_format,
    pad_image=True,
    initial_image_shape=[1280, 1918],
    target_image_shape=[1280, 1920],
    batch_size=1,
    num_parallel_batches=8,
    num_prefetch=None):
    
    image_filename_tensor = tf.data.Dataset.from_tensor_slices(image_filenames)
    images = image_filename_tensor.apply(tf.contrib.data.map_and_batch(
        lambda image: parser(image, None, 
            data_format,
            pad_image=pad_image,
            initial_image_shape=initial_image_shape,
            target_image_shape=target_image_shape),
        batch_size=batch_size,
        num_parallel_batches=num_parallel_batches))

    images = images.prefetch(num_prefetch)
    return images

## Old versions:
## These versions have been kept only for reference. They read data from TFRecord files.

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

# def input_fn(filename, image_shape, data_format, train, num_repeat=1, batch_size=1):
#     # Training Performance: A user's guide to converge faster (TF Dev Summit 2018)
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
