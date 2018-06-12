# drive-ai
## Coding A Self-Driving Car AI (Using Behavioral Cloning To Learn From Human Examples)

Watch a video demonstration of the trained model [here] (https://youtu.be/yjsqpEfjWIo)

The goals / steps of this project are the following:

* Use the Udacity car-simulator to collect images representing desired driving behavior
* Build a convolutional neural network architecture in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set (see [model.py](./model.py))
* Test that the model successfully drives around a track autonomously without leaving the road

****

[//]: # (Image References)

[image1]: ./examples/01.jpg "Centre Driving"
[image2]: ./examples/02.jpg "Recovery - side"
[image3]: ./examples/03.jpg "Recovery - centre"
[image4]: ./examples/04.png "Training and Validation"

---
### Files and Code

#### 1. This repository contains code that can be used to train a model capable of driving a car in a simulator in autonomous mode

My project includes the following files:

* model.py - containing the pipeline used to create, train and validate the model
* README.md - this file

#### 2. Running this code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
where model.h5 is a saved convolutional neural network created using model.py.

### Model Architecture and Training Strategy

#### 1. Model architecture employed

My model consists of a convolution neural network, based on the [Nvidia end-to-end architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 

It also starts with a normalisation layer (using a Keras lambda layer in code line 125) which also mean centers the data. For more detail on the final architecture employed, see below.

The model includes RELU layers to introduce nonlinearity (code lines 132, 134, 136, 138, 140, 144, 146, 148, 150).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer before the first fully connected layer in order to reduce overfitting (model.py line 142). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 46-51). The model was tested by running it through the simulator and ensuring that the vehicle would stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 156).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving along the track in the reverse direction and driving focussing on smooth turns. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple and add model complexity / power. Additional data was collected for areas of the track where the model performed poorly, and the model retrained to incorporate this additional data.

My first step was to use a convolutional neural network model similar to the LeNet architecture. I used this model as a simple baseline because it has been proven in character recognition, but is relatively small and quick to train. I changed the architecture to the Nvidia architecture when I had confidence that my training pipeline was working.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Although the car was able to navigate the track, it drove with sudden, sharp turns.

To mitigate overfitting, I modified the model to add a dropout layer prior to the first fully connected layer. 

The final step was to run the simulator to see how well the car was driving around track one, and iteratively refine the model and capture more training data.

At the end of the process, the model is capable of driving the vehicle autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 130-151) consisted of a convolutional neural network with the following layers and layer sizes:

* a normalisation and mean-centering layer: Input = 640x480x3. Output = 640x480x3
* a layer to crop the image (75 pixels from the top and 25 from the bottom) in order to remove areas of the image that may confuse the network: Output = 640x380x3
* a convolutional layer with 24 filters of size 5x5 and 'relu' activation: Output = 318x188x24
* a convolutional layer with 36 filters of size 5x5 and 'relu' activation: Output = 157x92x36
* a convolutional layer with 48 filters of size 5x5 and 'relu' activation: Output = 77x44x48
* a convolutional layer with 64 filters of size 3x3 and 'relu' activation: Output = 75x42x64
* a convolutional layer with 64 filters of size 3x3 and 'relu' activation: Output = 73x40x64
* A dropout layer with probability = 0.2
* Fully connected layer with 'relu' activation. Output = 1164
* Fully connected layer with 'relu' activation. Output = 100
* Fully connected layer with 'relu' activation. Output = 10
* Fully connected layer with 'relu' activation. Output = 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I combined the provided sample data with additional captured data. I recorded recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover should it go off course. These images show what a recovery looks like starting from the car at the right side of the bridge:

![alt text][image2]
![alt text][image3]

To augment the dataset, I also flipped images and angles in order to reduce the likelihood of the model overfitting. 

After the collection process, I had 190,968 number of data points. I then preprocessed this data by normalising, mean centering and cropping the top and bottom of the images (75 pixels from the top and 25 from the bottom).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. A visualisation of the training and validation loss of the model is (both can be seen to be decreasing monotonically).

![alt text][image4]

I used an adam optimizer so manually tuning the learning rate was not necessary.

Watch a video demonstration of the trained model [here] (https://youtu.be/yjsqpEfjWIo)

