# Fashion-Recommendation-System
This is ML model for the task of fashion recommender system

Pre-requisites for this project are-
1. DeepLearning(Secifically CNN)
2. Transfer Learning- as the model we will be using is ResNET and the data set used will be imageNET(which is a large visual dataset with 14million images and its labelled)


SOME FACTS ABOUT CNN-
   1. why we ain't using ANN(Aritificial Neural Networks)?
       Because if we take an example of 600x400 pixel image i.e. 240000 pixels and if we create closed entry points and a input layer of 128 inputs that would result in a large number of weigths (approx 2cr). The problem with this would be
a. overfitting
b. loss of important information like sapcial arrangement(what's where)

![image](https://github.com/user-attachments/assets/a4182a95-82dc-4066-8e3b-f0dbe1bf0b81)
In this image the various convolutional layers work same as the human mind the initial layers keep on extracting the features of the image and at the end the final image is recognised
Look about- (convolutional operations, (n-m+1))

Some Points about IMAGENET:
1. Has 14million images
2. There were competitons organised by ImageNet wherein the lowest error percentage was to be achieved in trainig a CNN model and along the years the VGG16 presented a model with an architecture that provided lowest error percentage.
3. Transfer learning is thus a tuning mechanism on the architecture of VGG16.
![image](https://github.com/user-attachments/assets/f1226311-b640-4adc-9ef5-84868df6050e)

