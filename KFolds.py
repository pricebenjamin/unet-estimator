import numpy as np

class KFolds():
    '''
    A class for constructing `k` non-overlapping, equally sized subsets of 
    images (i.e. folds). 

    For the Carvana dataset, images are first paired with their mask, then 
    grouped by car. In total, there are 318 different cars in the dataset. Each 
    car is then randomly assigned into one of the `k` folds.

    Currently, this class requires that the number of examples (e.g. cars)
    is evenly divisible by the number of folds.
    '''
    def __init__(self, image_filenames, mask_filenames, num_folds, \
        sort=False, yield_dict=False, random_seed=42):

        self.yield_dict = yield_dict

        num_images = len(image_filenames)
        assert num_images == len(mask_filenames)

        # Assumptions are written in UPPERCASE
        NUM_IMAGES_PER_CAR = 16
        assert num_images % NUM_IMAGES_PER_CAR == 0
        num_cars = num_images // NUM_IMAGES_PER_CAR

        self.num_folds = num_folds
        assert num_cars % num_folds == 0 # TODO: How should we define folds if
        # the number of examples is not evenly divisible by num_folds? Should
        # we throw away the remainder or have different sized folds?

        # Zip images with their corresponding masks.
        if sort:
            image_filenames = sorted(image_filenames)
            mask_filenames = sorted(mask_filenames)

        self.pairs = list(zip(image_filenames, mask_filenames))
        self.pairs = np.array(self.pairs)

        # Group images of the same car into separate sets.
        # Note: Since the filenames are sorted, this can be accomplished by
        # reshaping the list of filenames.
        self.pairs = self.pairs.reshape((num_cars, NUM_IMAGES_PER_CAR, 2))

        # Construct a reproducible random state:
        r = np.random.RandomState(seed=random_seed)
        self.fold_assignment = r.permutation(range(num_cars)) % self.num_folds

    def __iter__(self):
        for i in range(self.num_folds):
            yield get_fold(i)

    def get_fold(self, n):
        assert isinstance(n, int)
        assert (n >= 0) and (n < self.num_folds)

        # Construct a boolean mask to select cars in the given fold.
        mask = (self.fold_assignment == n)

        # Note that train and evaluation pairs are complementary.
        eval_pairs = self.pairs[mask].reshape((-1, 2)).tolist()
        train_pairs = self.pairs[~mask].reshape((-1, 2)).tolist()

        # Unzip the pairs back into image filenames and mask filenames.
        eval_images, eval_masks = map(list, zip(*eval_pairs))
        train_images, train_masks = map(list, zip(*train_pairs))

        if self.yield_dict:
            return dict(
                training=dict(images=train_images, masks=train_masks),
                evaluation=dict(images=eval_images, masks=eval_masks))
        else:
            return (train_images, train_masks), (eval_images, eval_masks)
