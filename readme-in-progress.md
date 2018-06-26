# Implementing U-Net using the TensorFlow Estimator API

This repository contains scripts for building, training, and generating predictions with our own implementation of the U-Net architecture. The network was implemented in Python using TensorFlow's Estimator API. Our goal was to reproduce the U-Net architecture described in [REF], then train the network on data provided by the Carvana Image Masking Challenge hosted on Kaggle. Our implementation produced masks with average pixel accuracy of [ACC] +/- [ERROR] over a 6-fold cross-validation set [CHECK TERMINOLOGY].

Example masks produced by a single network trained on one fold of the cross-validation set can be seen below. 

[EXAMPLE MASKS]

### Description of the original U-Net architecture

U-Net is a deep, fully convolutional neural network architecture proposed for biomedical image segmentation. A visual representation of the network, as shown in the original publication [REF], can be found below.

[IMAGE OF UNET]

Following along with the image given above, a 3-channel (RGB) image of size $h \times w$ enters the network on the left side. A two-dimensional convolution operation, consisting of 32 separate 3x3 kernels, is then applied to the image, resulting in a 32-channel image of size $(h-2) \times (w-2)$ [FOOTNOTE-1]. The resulting 32-channel image is then fed into another convolution operation...

a two-class probability distribution?

[FOOTNOTE-1]: The reduction in height and width is the result of applying a `padding='valid'` convolution, which prevents kernels from moving beyond the edges of the input image. The common alternative to this is a `padding='same'` convolution, which pads the edges of the input image with zeros such that the output of the operation has the same height and width as the input.
### Differences between our implementation and the original architecture

### Summaries of each file

### Running the program
