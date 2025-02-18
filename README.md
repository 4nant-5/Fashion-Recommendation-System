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
1. Has 14million labelled images
2. There were competitons organised by ImageNet wherein the lowest error percentage was to be achieved in trainig a CNN model and along the years the VGG16 presented a model with an architecture that provided lowest error percentage.
3. Transfer learning is thus a tuning mechanism on the architecture of VGG16.
![image](https://github.com/user-attachments/assets/f1226311-b640-4adc-9ef5-84868df6050e)

What do you do in transfer learning?
In trasnfer learning intread of making ones own model we use a different well trained. But this sparks two questions-
1. Why shouldn't I make my own model?
2. How will the other model work onmy data?
Because-
1. Lack of data, since deep learning model is data hungry and providing such big labelled dataset is not easy.
2. Apart from this almost every problem will arive such as overfitting, etc.
3. Also, althought imageNET doen't contain our data but it has already seen thousands of features classes and analysed a pattern, thus it gets easier to apply that pattern on our data set as well.

How we make use of transfer learning?(A walkthrough)
An example- u take longer to learn how to ride a bicycle than ride a motorcycle, its so because when you learn to ride a bike you already learn some features such as balancing, road sense,etc. While riding a bike you don't have to think about the skills during bicycle but you learn newer skills like gear control, etc.

Similarly, as in a CNN model the last part which identifies the image by summing up the features we just remove that particular part and replace it with a operation required for our model. This however, is not necessary if you have a lot of data and a sufficient computational power you may train more than one layer of data.
