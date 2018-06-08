from PIL import Image
from multiprocessing import Pool

from train_estimator import model_dir, eval_tfrecord, image_shape, num_channels, \
                            model_fn, input_fn

import tensorflow as tf
import numpy as np
import os

cutout_dir = os.path.expanduser('~/repos/unet-estimator/predictions/small/cutouts')

def save_cutout(enum_prediction):
    i, prediction = enum_prediction # Unpack the list
    if i % 100 == 0:
        print('Saving image {}'.format(i))
    cutout = prediction['cut']
    cutout = cutout.astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout_filename = os.path.join(cutout_dir, str(i) + '.png')
    cutout.save(cutout_filename)
    # type(prediction['cut']) == <class 'numpy.ndarray'>
    # prediction['cut'].shape == (1280, 1920, 3)
    # prediction['cut'].dtype == float32


def main():
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
 
    params = {
        'image_shape': image_shape,
        'num_channels': num_channels,
        'data_format': data_format 
    }
 
    # Initialize the Estimator
     image_segmentor = tf.estimator.Estimator(
         model_dir=model_dir,
         model_fn=model_fn,
         params=params)
 
     print('Computing predictions...')
     predictions = image_segmentor.predict(
         input_fn=lambda: input_fn(
             eval_tfrecord,
             image_shape,
             data_format,
             train=False))
 
     # Save cutout images.
     # multiprocessing.Pool is used to save multiple images at a time.
     with Pool(processes=int(os.cpu_count()/2)) as p:
         p.map(save_cutout, enumerate(predictions))

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
