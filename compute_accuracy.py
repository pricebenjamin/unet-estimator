# This file is meant to store the cross-validation accuracies
# of various iterations of the Unet implementation.

# These accuracies are usually manually extracted from slurm 
# output files located in ~/slurms/outputs. The specific file-
# name is provided above each list.

import numpy as np

# Files: ~/slurms/outputs/standard-unet/folds012.out,
#        ~/slurms/outputs/standard-unet/folds345.out
# Note: 'global_step': 44520
# Note: KFold generator fed with random_seed = 42
accuracy = [
    0.9960292,
    0.99600857,
    0.99620098,
    0.99547887,
    0.99558622,
    0.99629545
]

n = len(accuracy)
x_bar = np.average(accuracy)

squared_dev = [(x - x_bar)**2 for x in accuracy]
sample_std_dev = np.sqrt(sum(squared_dev)/(n-1))

print('Accuracy: {} +/- {}'.format(x_bar, sample_std_dev))

