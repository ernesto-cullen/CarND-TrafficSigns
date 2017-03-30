# **Traffic Sign Recognition** 

## Self-driving car Nanodegree


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./dataset_sample.png "Dataset sample"
[image1]: ./max80_12591.png "Max 80 - Sample #12591"
[image2]: ./sample_before_normalizing.png "Sample image before normalization"
[image3]: ./sample_after_normalizing.png "Sample image after normalization"
[image4]: ./examples/children_crossing.jpg "Children Crossing"
[image5]: ./examples/max60.jpg "Max60"
[image6]: ./examples/no_entry.jpg "No Entry"
[image7]: ./examples/no_passing_over_tons.jpg "No passing for vehicules over 15 Tn"
[image8]: ./examples/pedestrians_with_back.jpg "Pedestrians"
[image9]: ./examples/roundabout_mandatory "Roundabout mandatory"
[image10]: ./examples/stop.jpg "Stop"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the readme/writeup, and here is a link to my [project code](https://github.com/ernesto-cullen/CarND-TrafficSigns/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The dataset consisted of 34799 images of german traffic signs with their corresponding labels for training, 4410 images for validation and 12630 images for testing.
Each image is 32x32 pixels, 3 channels (RGB).
There are 43 classes in the dataset, with an uneven distribution:

|Code|Description|# of images|
|---|---|---:|
| 0 | Speed limit (20km/h) | 180 |
| 1 | Speed limit (30km/h) | 1980 |
| 2 | Speed limit (50km/h) | 2010 |
| 3 | Speed limit (60km/h) | 1260 |
| 4 | Speed limit (70km/h) | 1770 |
| 5 | Speed limit (80km/h) | 1650 |
| 6 | End of speed limit (80km/h) | 360 |
| 7 | Speed limit (100km/h) | 1290 |
| 8 | Speed limit (120km/h) | 1260 |
| 9 | No passing | 1320 |
| 10 | No passing for vehicles over 3.5 metric tons | 1800 |
| 11 | Right-of-way at the next intersection | 1170 |
| 12 | Priority road | 1890 |
| 13 | Yield | 1920 |
| 14 | Stop | 690 |
| 15 | No vehicles | 540 |
| 16 | Vehicles over 3.5 metric tons prohibited | 360 |
| 17 | No entry | 990 |
| 18 | General caution | 1080 |
| 19 | Dangerous curve to the left | 180 |
| 20 | Dangerous curve to the right | 300 |
| 21 | Double curve | 270 |
| 22 | Bumpy road | 330 |
| 23 | Slippery road | 450 |
| 24 | Road narrows on the right | 240 |
| 25 | Road work | 1350 |
| 26 | Traffic signals | 540 |
| 27 | Pedestrians | 210 |
| 28 | Children crossing | 480 |
| 29 | Bicycles crossing | 240 |
| 30 | Beware of ice/snow | 390 |
| 31 | Wild animals crossing | 690 |
| 32 | End of all speed and passing limits | 210 |
| 33 | Turn right ahead | 599 |
| 34 | Turn left ahead | 360 |
| 35 | Ahead only | 1080 |
| 36 | Go straight or right | 330 |
| 37 | Go straight or left | 180 |
| 38 | Keep right | 1860 |
| 39 | Keep left | 270 |
| 40 | Roundabout mandatory | 300 |
| 41 | End of no passing | 210 |
| 42 | End of no passing by vehicles over 3.5 metric tons | 210 |
Total:  34799

Below is an image of 10 samples for each class

![samples][image0]

We can see there are images where the sign is clearly visible while others are barely recognizable even for a human. Also, there are large variations on illumination, contrast, brightness, etc. This is necessary to allow the network to generalize.


#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

For the purposes of this project, we can work with grayscale images as the color is not the main feature. The original dataset was converted to grayscale as a preprocessing step, along with an augmentation -discussed in next section.
This allows me to train the network with more data in less time, without the need of a big CUDA machine.
All processing was done in my personal laptop, using a NVidia GeForce GTX 960M with following characteristics:

* CUDA cores: 640
* Memory: 2GB GDDR5 5010MHz
* Base clock: 1097 MHz

The notebook contains code to display a single image and a vector of images in Cell 3. A more complete visualization is in notebook "image_visualization.ipynb".


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I preprocessed the training set by augmenting it generating two copies of each image, one rotated +15 degrees, the other -10 degrees. Also converted each image to grayscale to speed the computation.
To further speed the process of trial and error, I saved the pre-processed data using pickle. The code to pre-process the images and save to the new files is in notebook "preprocess.ipynb". In the main notebook I just loaded the preprocessed files.

Here is an example of a traffic sign image before and after grayscaling, including its two rotations: +15 degrees and -10 degrees

![train sample][image1]

As a last step, I normalized the image data around 0 to get a better pixel value distribution. Here is a plot of the same image before and after the normalization, to visually validate the content is intact:

|Before   | After  |
|:-------:|:------:|
![sample before normalization][image2] | ![sample after normalization][image3]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data provided was already separated in training and validation sets. The only processing I did was the augmentation of training set by rotating the images as explained above.

My final training set had 104397 images.



#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 'Model architecture' section of the notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image 						| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
|||
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
|||
| Fully connected		| 300 units    									|
| RELU					|												|
| Dropout				| 												|
|||
| Fully connected		| 200 units    									|
| RELU					|												|
| Dropout				| 												|
|||
| Fully connected		| 43 units    									|
 



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the section 'Train, Validate and Test the Model' of the notebook.

To train the model, I used 15 epochs with a batch size of 256. These values are set in the notebook before defining the network.
The training uses an Adam optimizer with a learning rate of 0.001. I tried different learning rates, bigger and smaller, and decided for this as the one that got the best results.
The training operation tries to minimize the loss, defined as the mean of the cross entropy of the net:

```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=signs, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
```
where *signs* is the output of the net, and *labels* is the one-hot encoding of the labels.


#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located after the training in the notebook. It uses a function *evaluate* which is called with the validation set after each training epoch:

```python
def evaluate(X_data, y_data):
    n = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, n, BATCH_SIZE):
        if offset + BATCH_SIZE > len(X_data):
            batch_x, batch_y = X_data[offset:], y_data[offset:]
        else:
            batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / n
```
This function compares the result of pass each image in the validation set through the net with the correct label, given with the set. The result is normalized to 0..1.


My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.963
* test set accuracy of 0.936

I arrived to these numbers after a process of trial and error. I started with the LeNet architecture we saw in the videos, and made a lot of changes to get a high accuracy.

- tried adding convolutional layers. Didn't improve in general, and the training times spiked up.
- tried with only one convolutional layer. The accuracy was lower, so I settled for two convolutional layers.
- tried several combinations of fully connected layers. The one selected give a good accuracy with not much extra time.
- added the dropout to get the test accuracy closer to the validation accuracy. Without this, the net was overfitting the training data, getting almost perfect accuracy on the training dataset and much lower on the validation dataset. I use a keep probability of 0.4 while training.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the Step 3 on the notebook

Here are the results of the prediction:

| Image			                                |     Prediction	        					| 
|:---------------------------------------------:|:---------------------------------------------:| 
| Children crossing		                        | General caution								| 
| Speed limit (60km/h)	                        | Speed limit (60km/h)							|
| No entry				                        | No entry										|
| No passing for vehicles over 3.5 metric tons	| No passing for vehicles over 3.5 metric tons	|
| Pedestrians                           		| Right-of-way at the next intersection     	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares disfavorably to the accuracy on the test set of 93.6%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model predicted wrongly a 'General Caution' sign. The correct sign 'Children crossing' was not even between the five most probable outcomes:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .993         			| General Caution								| 
| .005     				| Right-of-way at the next intersection     	|
| .0004					| Beware of ice/snow							|
| .00002      			| Traffic signals				 				|
| .00001			    | Slippery road      							|

Looking at the feature maps for the first conv layer on this image, we see that the basic shape is recognized (triangle), but this is also a salient feature of the 'General Caution' sign and 'Right-of-way at the next intersection' so the problem here is the level of detail inside the triangle. I tried with a 3x3 patch size in the second conv layer, but the image was still not recognized and the net didn't recognize the second image (speed limit 60 km/h) so I went back to 5x5 in the second layer too.


For the second image, the model recognized correctly the Speed Limit (60km/h) sign with a high confidence (0.9999) while the other possibilities are much smaller:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999        			| Speed limit (60km/h)							| 
| 8.717e-7      		| Children crossing                          	|
| 1.807e-8				| Keep right        							|
| 6.068e-11      		| Turn left ahead				 				|
| 4.175e-11			    | Speed limit (30km/h)							|

interestingly, the only other 'speed limit' proposed by the net was the one for 30 km/h with a very low probability. I would have expected to get more 'speed limit' signs.


The third image was also recognized correctly ('No entry' sign). No surprises here, the signal is pretty simple with easy to recognize features.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999        			| No entry          							| 
| 2.924e-7      		| Stop                                        	|
| 9.050e-9				| Roundabout mandatory 							|
| 7.227e-9      		| Keep right    				 				|
| 5.562e-9			    | Turn left ahead   							|


The fourth image ('No passing for vehicles over 3.5 metric tons') was also recognized successfully with almost complete confidence: the other proposals combined amount to a probability of 1.0541841035e-9.

| Probability         	|     Prediction	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| 1.000        			| No passing for vehicles over 3.5 metric tons	        | 
| 8.783e-10      		| Turn right ahead                           	        |
| 1.420e-10				| Speed limit (80km/h) 							        |
| 2.823e-11     		| End of no passing by vehicles over 3.5 metric tons    |
| 5.672e-12			    | Ahead only                   							|


The last image was not recognized correctly: the correct label is 'Pedestrians' but the net proposed 'Right-of-way at the next intersection'. The sign was chosen because it is a variation of the sign in the training set, with a 'zebra path'. This was enough to confuse the net to produce these outcomes:

| Probability         	|     Prediction	        			        | 
|:---------------------:|:---------------------------------------------:| 
| 0.9986       			| Right-of-way at the next intersection	        | 
| 5.684e-04      		| Beware of ice/snow                   	        |
| 5.539e-04				| Roundabout mandatory 					        |
| 2.362e-04     		| Dangerous curve to the right                  |
| 6.408e-05			    | Slippery road        							|

The correct sign was not in the top 5 proposals.

