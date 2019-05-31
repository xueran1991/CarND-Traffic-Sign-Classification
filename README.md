# **Self-Driving Car Engineer** Nanodegree
## CarND-Traffic-Sign-Classification

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. The data was restore from the pickle file. Below is a basic summary of the data set. The original data has already been split into train,validaton and test these 3 parts.

You can download the dataset [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip).

    Number of training examples = 34799 
    
    Number of validation examples = 4410
    
    Number of testing examples = 12630
    
    Image data shape = (32, 32, 3)
    
    Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here are some examples of the dataset. Through the histagrom of the data, we can see the amount of the samples of different labels varied from about 250 to 2000.

<img src="for writeup/examples of data.png">

<img src="for writeup/data distribution.png">

### Design and Test a Model Architecture

#### 1. Preprocessing the image data. 

As a first step, I decided to convert the images to grayscale because the colors don't contain much information of what these traffic signs are and after converting to grayscale, the architecture will be less parameters. Hence our model will be less computationally-expensive.


Then I normalize the dataset. After normalization, the mean of the train set data is about -0.36, and all the pixels' value from -1 to 1. Normalization makes the data distribution more even. It helps the error gradient decending  works well.

I decided to generate additional data because some of the class contains as less as 250 pictures. It's few for training. I randomly warping,rotate and zoom the original pictures. And append them to the original dataset. After the data augmentation, each class has 3000 samples.

Here are some pictures of before and after the augmentation.

<img src="for writeup/data augmentation.png">

#### 2. Final model architecture.
 My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 3x3x128 	|
| RELU					|												|
| Dropout               | keep_prob=0.2                                 |
| Fully connected		| W(1152, 200), b(200) 							|
| Fully connected		| W(200, 43), b(43)		    					|
| Softmax				|												|



#### 3. Training process.

At first I use the LeNet5 as my original training model. But I couldn't get a higher accuracy than 90% on the valid set. So I change my model stucture. I added more convolutional layer. And change the filter size from 5x5 to 3x3. Recent popular nerual network architectures suggest the use of  little filter size and more filters.

To train the model, I used cross entropy as the lost function and Adam as the optimizer. I also use the changing learning rate from the original 0.001. The the learning rate will decay through the epochs.

I tune the dropout ratio for many times for 0.1 to 0.5. At the keep_prob=0.2, I got the highest accuracy score. The plots show training process with differernt parameters.

<img src="for writeup/acc-dp=0.5.png">
<img src="for writeup/acc-dp=0.3.png">
<img src="for writeup/acc-dp=0.2.png">

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.984
* test set accuracy of 0.974

I choose The LeNet5 as my first architecture. Beacause it have achieved high sore on MNIST dataset. The architecture contains only 2 convolitional layers, 2 subsampling layers and 2 fully-connected layer. It's easy to train.

One of the shortage of the architectue is that when the traing set is not large enough, it's hard to get a high accuracy. So I add more convolutional layers to the architecture. So there are more features about the image in my model.

There may some problem with the augmentation. I am not sure which validation set to use when training the model. I use the original validation set. The accuracy became vary low on the training set, but after traing the accruacy on the test set  didn't change much.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I download from the GTSRB  offical website.

<img src="for writeup/new images.png">

The first image might be difficult to classify because it's so blurry that is difficult for us to see whatis inside the triangle.

The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. It's a pretty good result.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The results are as the pictures blow. We can see that the modle isn't 100% sure about what those pictures are. But the correct always have the highest probilities.
<img src="for writeup/topk-1.png">
<img src="for writeup/topk-2.png">
<img src="for writeup/topk-3.png">
<img src="for writeup/topk-4.png">
<img src="for writeup/topk-5.png">

I thought the first image was the most diffcult to classify. But the model get more confused on the Stop sign.




