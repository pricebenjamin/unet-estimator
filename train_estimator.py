# Local imports
from input_fn import input_fn
from KFolds import KFolds
from model import model_fn

# Load external modules
from glob import glob
import tensorflow as tf
import os
import argparse

LEARNING_RATE = 1e-5

# TODO: Accept commandline arguments for the following variables
MODEL_DIR = '/mnt/lfs2/pric7208/tf-saves/cross-validate/standard-unet/fold'
NUM_EPOCHS = 20
EPOCHS_BETWEEN_EVALS = 2

WORKING_DIR = '/mnt/lfs2/pric7208/kaggle/carvana'
IMAGE_DIR = os.path.join(WORKING_DIR, 'train_hq')
MASK_DIR  = os.path.join(WORKING_DIR, 'train_masks')

IMAGE_FILENAMES = sorted(glob(os.path.join(IMAGE_DIR, '*.jpg')))
MASK_FILENAMES = sorted(glob(os.path.join(MASK_DIR, '*.gif')))
NUM_FOLDS = 6

NUM_OUTPUT_CLASSES = 2
# Pixels are classified as either "foreground" or "background"

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 4

# TODO: Which command line args are passed to main?
def main(fold_nums):
    # Check if the system's version of TensorFlow was built with CUDA (i.e. uses a GPU)
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    params = {
        'data_format': data_format,
        'num_output_classes': NUM_OUTPUT_CLASSES,
        'learning_rate': LEARNING_RATE
    }

    # Mirror the model accross all available GPUs using the mirrored distribution strategy.
    # TODO: Enable multi-gpu support via commandline arg instead of by default
    # - Is there a good way to check if multiple GPUs are available?
    distribution = tf.contrib.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(
        train_distribute=distribution,
        keep_checkpoint_max=2,
        log_step_count_steps=5)

    folds = KFolds(IMAGE_FILENAMES, MASK_FILENAMES, 
        num_folds=NUM_FOLDS, sort=False, yield_dict=False)

    # Train separate models on each requested fold.
    for fold_num in fold_nums:
        (train_images, train_masks), (eval_images, eval_masks) = folds.get_fold(fold_num)

        # Initialize the Estimator
        image_segmentor = tf.estimator.Estimator(
            model_dir='-'.join([MODEL_DIR, str(fold_num)]),
            model_fn=model_fn,
            params=params,
            config=config)

        # Train and evaluate
        for i in range(NUM_EPOCHS // EPOCHS_BETWEEN_EVALS):
            print('\nEntering training epoch %d.\n' % (i * EPOCHS_BETWEEN_EVALS))
            image_segmentor.train(
                # input_fn is expected to take no arguments
                input_fn=lambda: input_fn(
                    train_images,
                    train_masks,
                    training=True,
                    data_format=params['data_format'],
                    num_repeats=EPOCHS_BETWEEN_EVALS,
                    batch_size=TRAIN_BATCH_SIZE))

            results = image_segmentor.evaluate(
                input_fn=lambda: input_fn(
                    eval_images,
                    eval_masks,
                    training=False,
                    data_format=params['data_format'],
                    batch_size=EVAL_BATCH_SIZE))

            # TODO: Look into writing example images to a tf.summary?
            print('\nEvaluation results:\n%s\n' % results)

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    # TODO: Update --folds argument to accept multiple integers.
    # This can be done with `nargs='+'`, I think...
    parser.add_argument('-f', '--folds', type=str, required=True,
        help='a string which can be evaluated to a python list; the python list '\
             'should contain integers from the interval [0, 5] indicating which '\
             'folds the network will use for training')
    # TODO: Consider adding flags for the model directory, data directory, number
    # of epochs, epochs between evals. Choose good defaults.
    args = parser.parse_args()
    
    assert args.folds[0] == '[' and args.folds[-1] == ']'
    fold_nums = eval(args.folds) # This is probably *very* unsafe.

    main(fold_nums)
