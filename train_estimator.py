# Local imports
from input_fn import input_fn
from KFolds import KFolds
from model import model_fn

# Load external modules
from glob import glob
import tensorflow as tf
import os
import argparse

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

    # TODO: Test argument parsing and validation
    # TODO: Move argument parsing to its own script?

    parser = argparse.ArgumentParser(
        # TODO: description, epilog, other?
        )
    
    # Directory arguments
    parser.add_argument('--model-dir', type=str, required=True,
        help='location in which to find and/or save the network model '\
             'and Tensorboard summaries; if cross-validation is used, '\
             'the given location will be used as the base name for '\
             'each network\'s folder and a fold index will be appended')
    # TODO: Make the above help statement more clear/concise
    parser.add_argument('--data-dir', type=str, required=True,
        help='location of Carvana Image Masking Competition data')

    # Cross-validation arguments
    parser.add_argument('--no-cv', action='store_true',
        help='do not use cross-validation when training (default: use cross-'\
             'validation)')
    parser.add_argument('-K', '--num-folds', metavar='k', type=int,
        help='an integer specifying how many bins (i.e. subsets or folds) '\
             'the training data will be split between')
    parser.add_argument('--folds', metavar='n', type=int, nargs='+',
        help='an integer (or list of integers) specifying which fold(s) to '\
             'train against. Fold indexes start at zero.')

    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=1,
        help='number of epochs over which the network will be trained')
    parser.add_argument('--epochs-between-evals', type=int,
        help='number of epochs to complete before running another evaluation '\
             '(uses --num-epochs by default, corresponding to a single '\
             'evaluation at the end of training)')
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--train-batch-size', type=int, default=1)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    # TODO: is this a reasonable default for eval batch size?
    parser.add_argument('--distribute', action='store_true',
        help='specifies whether or not to use distributed training (default: '\
             'do not use distributed training)')

    # TODO: --save-best; do not save the model unless its accuracy is better
    # than the previously saved model. This might be an option somewhere in
    # the Estimator API.
    
    args = parser.parse_args()

    # Verify training arguments
    if args.num_epochs <= 0:
        raise parser.error('--num-epochs must be a positive number')
    if args.epochs_between_evals is None:
        args.epochs_between_evals = args.num_epochs
    if args.epochs_between_evals > args.num_epochs:
        raise parser.error('--epochs-between-evals must be less than or '\
            'equal to --num-epochs')

    # Verify cross-validation arguments
    if args.no_cv:
        # TODO: Allow --no-cv
        raise NotImplementedError('Cross-validation is required for now.')
        # User has specified that cross-validation should not be used;
        # check that this does not conflict with the remaining arguments.
        if args.num_folds is not None:
            raise parser.error('Contradictory flags --no-cv and --num-folds')
        if args.folds is not None:
            raise parser.error('Contradictory flags --no-cv and --folds')
    else:
        # Assume cross-validation; make sure that necessary flags are provided.
        if args.num_folds is None:
            raise parser.error('K-fold cross-validation is used by default; '\
                'please specify a number of folds to use.')
        if args.folds is None:
            # We interpret this to mean that we should train against all folds
            args.folds = list(range(args.num_folds))
        if min(args.folds) < 0 or max(args.folds) > args.num_folds - 1:
            raise parser.error('Fold indexes must be non-negative and less '\
                'than --num-folds.')

    main(args)
