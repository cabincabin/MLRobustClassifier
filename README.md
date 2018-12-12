# CS 539: Machine Learning - Final Team Project

## Development of Robust Image Classifiers for Geo-diverse Distributions

<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/inclusive_images_header.png"/>
</p>

### The Team
- Alexander Shoop (akshoop@wpi.edu, @akshoop)
- Claire Danaher (cedanaher@wpi.edu, @claireedanaher)
- Clayton Dembski (cjdembski@wpi.edu, @cabincabin)
- Saina Rezvani (srezvani@wpi.edu, @SainaRez)

### Introduction
The effectiveness of modern machine learning image classifiers is heavily dependent on the degree to which the corpus is representative of the images being classified. When corpuses are not inclusive, models produce high rates of misclassification of images with low representation. Previous research completed on in this area include an exploration of gender<sup>[1]</sup> and geodiversity<sup>[2]</sup> related challenges. 

This team project focused on the development of robust image input to improve classification of machine learning models to handle pictures from geographically diverse regions (primarily non-Americas and non-European). The inspiration for this project came from a Kaggle competition called the [Inclusive Images Challenge](https://www.kaggle.com/c/inclusive-images-challenge). This is where information on the training and test image datasets can be found as well.

The team developed a traditional Convolutional Neural Network (CNN) using Tensorflow and Keras. We also had to learn and utilize the Google Cloud Platform (GCP) services; the GCP services used were Cloud Storage and Model Training due to the enormous size of the Open Images dataset. The team hypothesized that emphasizing the structure of the input photos (aka. features such as edges) would allow for better generalization of models, allowing for an increased classification rate. The results of this project work suggest that a method that augments the dataset to support generalization is likely a better approach.

### The Data
The data used to complete this research project is part of the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html). As this is a very large dataset totaling over 500GB,for the classifyable data, our team had to use Google Cloud Platform with TensorFlow when implementing the models. We had 3 sets from this data, which we worked with: one of 200,000+ images and 500 trainable labels, one of 37000 images, made from 8 of the most common labels, and one of 3600 images, and the same 8 labels (for quick tests). "Train/Test" split the eurocentric data into training and testing data. "Train/Tune" was tested using the real-world test dataset, 1000 images across all 500 labels, gathered from the [Kaggle competition page](https://www.kaggle.com/c/inclusive-images-challenge/data).


### Image Processing
In our project, we have two sets of images. One set is the grey scale version of images and one set is only object edges in the images. We cropped the edge images and gray scale imagesto the given bounding boxes and resized them to 256 x 256 after. Our motivation for finding the edges was to minimize information to decrease bias and generalize the images. In this way, we would focus purely on the shape of the objects passed into the model, and not the tonallity. For example, with facial features, skin tone would not play a part. 
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/full1.jpg"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/edge1.jpg"/>
</p>

### Model
##### Preprocessing:
In order to run the model, first we perform some preprocessing steps. The first step is extracting the training and testing data from the dictionary. We split the data and dedicate 25% of the data to testing and 75% to training. We reshape the data to only have the grayscale channel. Then we convert both datasets to type float. Finally we normalize the data to get values between 0 and 1.

##### Creating and training the model:
After the prerprocessing steps, we create the model as a sequence of layers. The first two input layers are 2D convolution
layers. Each creates a convolution kernel that convolves with the layer input to create the output. We used 32 filters for each convolution layer, a 3 by 3 convolution kernel and the relu activation function.
Our next input layer is MaxPooling2D which is used to reduce the number of parameters in the model. It uses a 2 by 2 sliding window and finds the maximum number of the 4 values.
The next input layer is Dropout which normalizes the model and that helps prevent overfitting.
We use Flatten as the next input layer to flatten the data into a 1D array.
Finally our last model is Dense which takes in the final number of outputs which corresponds to the number of classes we use for classification. For this layer we use the softmax activation function to map inputs between 0 and 1.
After adding all the input layers to our model, we compile the model with the sparse_categorical_crossentropy loss function and the adam optimizer. Based on the graph below adam performs better than the rest.
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Optimizers_graph.png" 
width="700" height="370"/>
</p>
Source: https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

In the next step we fit the model to our training data with 10 epochs to train it for and we evaluate the training accuracy to see how well the model is fit to the training data. We also use callbacks to understand the internal state of the model while it is traning on data. 

### Results
For the two main training sessions -- black and white images, and edge images -- the following loss graphs were made. 
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Training_Loss_ Full_Image.png" width="400" height="370"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Training_Loss_ Edge_Image.png" width="400" height="370"/>
</p>
These graphs are plotted from the instantanious loss calculated by KERAS, sampled every 500 images. Each batch per epoch used for training was a bootstrapped sample of 10,000 images from the dataset. Each "Step" in the loss graph corrisponds to the beginning of one of these batches. 

For the results of our model, we split our training data into a training set, and a tuning set, then looked at the 
8 following labels:

      Person, Table, Tree, Building, Glasses, Boat, Insect, Dog

<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_test_full.png"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_test_edges.png"/>
</p>

Additionally, we trained a second model without the split, then tested on the 1000 non-eurocentric images provided by kaggle, 
again against the above 8 labels.

<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_tune_full.png"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_tune_edges.png"/>
</p>

### Discussion and Conclusion

While we bootstrapped images for both our tuning and our testing datasets, we were able to choose our method and "seed" for bootstrapping the data. Because of this, the data results from each of the 4 above tests were gathered from models trained in the exact same way. Because of this, we get to directly compare the test results with one another, assured that each model was trained in the same way and in the same order. 
In addition, each label has a separate number of images associated with it. One label could corrispond to 20,000 of the images used, while another could corrispond to only 10 of the images. This is a problem, as it means that, by training a label with proportionally more images, we could potentially skew the guesses of our label, so it would only guess what the model has seen the most. We, however, could not limit the number of images-per-label to the label with the smallest number of images, as the smallest label had magnitudes smaller images associatd with it, only 2 images. To solve this, we first examined the number of images per label, and realized it was a left skewed normal distribution (SEE FOLDER OTHER). Because of this, we chose the number of images per label to be the average of the mean and median of the distribution, at just over 4000 images per label. 


### To Run
1. Create a GCP account and a new project.
2. Follow the Readme in the folder PreprocessLabels
3. Follow the Readme in the folder Image PreProcessing
4. Follow the Readme in the folder MLEngineTrainers

### References
##### Academic Papers:
1. Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2017). Men also like shopping: Reducing gender bias amplification using corpus-level constraints. arXiv preprint arXiv:1707.09457.
2. Shankar, S., Halpern, Y., Breck, E., Atwood, J., Wilson, J., & Sculley, D. (2017). No Classification without Representation: Assessing Geodiversity Issues in Open Data Sets for the Developing World. arXiv preprint arXiv:1711.08536.
##### Other Sources:
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
