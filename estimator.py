# Local imports
from input_fn import input_fn
from KFolds import KFolds
from model import model_fn
from arg_parser import arg_parser

# Load external modules
from glob import glob
import tensorflow as tf
import os
import sys

def main(args):
    
    # TODO: Should these variables still be capitalized?
    MODEL_DIR = args.model_dir
    WORKING_DIR = args.data_dir

    NUM_EPOCHS = args.num_epochs
    EPOCHS_BETWEEN_EVALS = args.epochs_between_evals
    LEARNING_RATE = args.learning_rate
    TRAIN_BATCH_SIZE = args.train_batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size

    NUM_FOLDS = args.num_folds
    FOLDS_TO_TRAIN_AGAINST = args.folds

    # Assumes default Carvana data folder structure...
    IMAGE_DIR = os.path.join(WORKING_DIR, 'train_hq')
    MASK_DIR  = os.path.join(WORKING_DIR, 'train_masks')

    IMAGE_FILENAMES = sorted(glob(os.path.join(IMAGE_DIR, '*.jpg')))
    MASK_FILENAMES = sorted(glob(os.path.join(MASK_DIR, '*.gif')))

    NUM_OUTPUT_CLASSES = 2
    # Pixels are classified as either "foreground" or "background"

    # Check if the system's version of TensorFlow was built with CUDA (i.e. uses a GPU)
    data_format = ('channels_first' if tf.test.is_built_with_cuda() \
        else 'channels_last')

    params = {
        'data_format': data_format,
        'num_output_classes': NUM_OUTPUT_CLASSES,
        'learning_rate': LEARNING_RATE
    }

    # Mirror the model accross all available GPUs using the mirrored distribution strategy.
    # TODO: Is there a good way to check if multiple GPUs are available?
    distribution = (tf.contrib.distribute.MirroredStrategy() if args.distribute\
        else None)
    config = tf.estimator.RunConfig(
        train_distribute=distribution,
        keep_checkpoint_max=2,
        log_step_count_steps=5)

    folds = KFolds(IMAGE_FILENAMES, MASK_FILENAMES, 
        num_folds=NUM_FOLDS, sort=False, yield_dict=False)

    # Train separate models on each requested fold.
    for fold_num in FOLDS_TO_TRAIN_AGAINST:
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

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = arg_parser(sys.argv)
    main(args)
