[//]: # (Image References)

[image1]: ./test_images/german_shepard1.jpg "German Shepard"
[image2]: ./vgg19-model.jpg "VGG19 Model Architecture"


## Dog Breed Classifier
!["=> 'This dog looks like a German shepherd dog'"][image1]
*=> 'This dog looks like a German shepherd dog'*

As part of my deep learning education in working with Convolutional Neural Networks (CNN), I used pretrained pytorch models and transfer learning to build an algorithim that is able to accurately classify images as containing humans, dogs, or neither, and in the first two cases predict the best resembling dog breed of the images subject.

The breed classifier model was trained on a dataset of 13000+ dog images labeled by breed and ran for 200 epochs with a 0.0025 learning rate. The model used a pretrained VGG19 model as it's base, with one custom fully connected layer inserted as its final linear layer to handle the classification of 133 different dog breeds. 

![VGG19 Architecture][image2]
*VGG19 Architecture*

Due to storage limitation, the training/datasets were not uploaded to this repository. Neither were the trained models saved during the project. However, the datasets can be downloaded at the following links:

- Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz). 


## Performance Snapshot

### Classifier model from Scratch (No transfer learning)

#### Model Architecture

```
Net(
  (conv1_1): Conv2d(3, 45, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv1_2): Conv2d(45, 45, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv1_3): Conv2d(45, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2_1): Conv2d(30, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2_2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2_3): Conv2d(10, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=24500, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=250, bias=True)
  (fc3): Linear(in_features=250, out_features=133, bias=True)
  (dropout): Dropout(p=0.2)
)
```

#### Hyperparams

- input image: 280 x 280 rgb
- learning rate: 0.025
- batch size: 20
- epochs: 200


#### Performance

- (START) Epoch: 1 Training Loss: 4.880341 Validation Loss: 4.872962
- (END) Epoch: 200 Training Loss: 3.189723 Validation Loss: 3.988435
- (TEST) Test Loss: 3.910456 Test Accuracy: 11% (100/836)


### Classifier model with VGG19 (transfer learning)

#### Model Architecture

```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=133, bias=True)
  )
)
```

#### Hyperparams

- input image: 224 x 224 rgb
- learning rate: 0.0025
- epochs: 100


#### Performance

- (START) Epoch: 1 Training Loss: 2.058879 Validation Loss: 1.254177
- (END) Epoch: 100 Training Loss: 0.446848 Validation Loss: 0.730253
- (TEST) Test Loss: 0.808622 Test Accuracy: 76% (643/836)