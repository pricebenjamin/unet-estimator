import argparse

def arg_parser(args_list):
    # TODO: Test argument parsing and validation
    # TODO: Move argument parsing to its own script?

    parser = argparse.ArgumentParser(
        # TODO: description, epilog, other?
        )
    
    # Estimator arguments
    # TODO: Write help comments
    # TODO: Should we allow users to specify more than one of these options?
    # TODO: Should --predict require its own data directory?
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--predict', action='store_true')

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
    
    args = parser.parse_args(args_list)

    # Verify Estimator arguments
    count_est_args = sum(map(int, [args.train, args.evaluate, args.predict]))
    estimator_options = '{--train, --evaluate, --predict}'
    if count_est_args > 1:
        raise parser.error('Please specify only one argument from {}.'.format(
            estimator_options)) # TODO: Consider alternative behavior
    if count_est_args == 0:
        raise parser.error('Please specify one argument from {}.'.format(
            estimator_options)) # TODO: Should we train by default?

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

    return args
