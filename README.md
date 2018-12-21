[//]: # (Image References)

[image1]: ./test_images/german_shepard1.jpg "German Shepard"
[image2]: ./images/vgg19-model.jpg "VGG19 Model Architecture"


## Dog Breed Classifier
!["=> 'This dog looks like a German shepherd dog'"][image1]

As part of my deep learning education in working with Convolutional Neural Networks (CNN), I used pretrained pytorch models and transfer learning to build an algorithim that is able to accurately classify images as containing humans, dogs, or neither, and in the first two cases predict the best resembling dog breed of the images subject.

The breed classifier model was trained on a dataset of 13000+ dog images labeled by breed and ran for 200 epochs with a 0.0025 learning rate. The model used a pretrained VGG19 model as it's base, with one custom fully connected layer inserted as its final linear layer to handle the classification of 133 different dog breeds. 

![VGG19 Architecture][image2]

Due to storage limitation, the training/datasets were not uploaded to this repository. Neither were the trained models saved during the project. However, the datasets can be downloaded at the following links:

- Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz). 
