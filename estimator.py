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

    # Assumes default Carvana data folder structure...
    # TODO: Consider pattern matching as mechanism for selecting
    # features vs labels?
    IMAGE_FILENAMES = sorted(glob(os.path.join(args.image_dir, '*.jpg')))
    MASK_FILENAMES = sorted(glob(os.path.join(args.mask_dir, '*.gif')))

    NUM_OUTPUT_CLASSES = 2
    # Pixels are classified as either "foreground" or "background"

    # Check if the system's version of TensorFlow was built with CUDA (i.e. uses a GPU)
    data_format = ('channels_first' if tf.test.is_built_with_cuda() \
        else 'channels_last')

    params = {
        'data_format': data_format,
        'num_output_classes': NUM_OUTPUT_CLASSES,
        'learning_rate': args.learning_rate
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
        num_folds=args.num_folds, sort=False, yield_dict=False)

    if args.train:
        # Train separate models on each requested fold.
        for fold_num in args.folds:
            (train_images, train_masks), (eval_images, eval_masks) = folds.get_fold(fold_num)

            # Initialize the Estimator
            image_segmentor = tf.estimator.Estimator(
                model_dir='-'.join([args.model_dir, str(fold_num)]),
                model_fn=model_fn,
                params=params,
                config=config)

            # Train and evaluate
            for i in range(args.num_epochs // args.epochs_between_evals):
                print('\nEntering training epoch %d.\n' % (i * args.epochs_between_evals))
                image_segmentor.train(
                    # input_fn is expected to take no arguments
                    input_fn=lambda: input_fn(
                        train_images,
                        train_masks,
                        training=True,
                        data_format=params['data_format'],
                        num_repeats=args.epochs_between_evals,
                        batch_size=args.train_batch_size))

                results = image_segmentor.evaluate(
                    input_fn=lambda: input_fn(
                        eval_images,
                        eval_masks,
                        training=False,
                        data_format=params['data_format'],
                        batch_size=args.eval_batch_size))

                # TODO: Look into writing example images to a tf.summary?
                print('\nEvaluation results:\n%s\n' % results)

    if args.evaluate:
        # TODO
        # How should the estimator be loaded in these subsections?
        # Which model will we use? Does it need to be specified
        # at the command line?
        pass

    if args.predict:
        # TODO
        pass

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = arg_parser(sys.argv[1:]) # Exclude program name
    main(args)
