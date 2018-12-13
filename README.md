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

## Website
https://cabincabin.github.io/MLRobustClassifier/

### Introduction
The effectiveness of modern machine learning image classifiers is heavily dependent on the degree to which the corpus is representative of the images being classified. When corpuses are not inclusive, models produce high rates of misclassification of images with low representation. Previous research completed on in this area include an exploration of gender<sup>[1]</sup> and geodiversity<sup>[2]</sup> related challenges. 

This team project focused on the development of robust image input to improve classification of machine learning models to handle pictures from geographically diverse regions (primarily non-Americas and non-European). The inspiration for this project came from a Kaggle competition called the [Inclusive Images Challenge](https://www.kaggle.com/c/inclusive-images-challenge). This is where information on the training and test image datasets can be found as well.

The team developed a traditional Convolutional Neural Network (CNN) using Tensorflow and Keras. We also had to learn and utilize the Google Cloud Platform (GCP) services; the GCP services used were Cloud Storage and Model Training due to the enormous size of the Open Images dataset. The team hypothesized that emphasizing the structure of the input photos (aka. features such as edges) would allow for better generalization of models, allowing for an increased classification rate. 

### The Data
The data used to complete this research project is part of the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html). As this is a very large dataset totaling over 500GB, for the classifiable data, our team had to use Google Cloud Platform with TensorFlow when implementing the models. We had 3 sets of this image data that we worked with: one of 200000+ images and 500 trainable labels, one of 37000 images, made from 8 of the most common labels, and one of 3600 images and the same 8 labels (for quick testing). The "Train/Test" dataset split the eurocentric data into training and testing data. "Train/Tune" was tested using the real-world test dataset, 1000 images across all 500 labels, gathered from the [Kaggle competition page](https://www.kaggle.com/c/inclusive-images-challenge/data).


### Image Processing
In our project, we have two sets of images. One set is the gray scale version of images and one set is only object edges in the images. We cropped the edge images and gray scale images to the given bounding boxes and resized them to 256 x 256 dimensions. Our motivation for finding the edges was to minimize information to decrease bias and to generalize the semantic meaning of the images. In this way, we would focus purely on the shape of the objects passed into the model, and not the tonality. For example, with facial features, skin tone would not play a part. 
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/full1.jpg"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/edge1.jpg"/>
</p>

### Model
##### Preprocessing:
In order to run the model, we first had to perform some preprocessing steps. The first step was to extract the training and testing data from the image dictionary. We split the dataset and dedicated 75% to training and 25% to testing. We reshaped the data to only have the grayscale channel. Then we converted both datasets to float datatype followed by normalizing the data.

##### Creating and training the model:
After the prerprocessing steps, we created the Keras CNN model as a sequence of layers. The first two input layers were 2D convolution
layers. Each created a convolution kernel that convolves with the layer input to create the output. We used 32 filters for each convolution layer, a 3 by 3 convolution kernel and the relu activation function.
Our next input layer was the MaxPooling2D layer which is used to reduce the number of parameters in the model. It used a 2 x 2 sliding window and finds the maximum number of the 4 values.
The next input layer was Dropout which normalized the model to help prevent the chances of overfitting to the training data.
We then flattened the layer into a 1D array.
Finally our last layer is the Denselayer which takes in the final number of outputs that corresponds to the number of classes we used for classification. For this last layer, we used the softmax activation function to map inputs between 0 and 1.
After adding all the input layers to our model, we compile the model with the sparse_categorical_crossentropy loss function and the adam optimizer. Based on the graph below adam performs better than the rest. Graph Source: https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Optimizers_graph.png" 
width="700" height="370"/>
</p>

In the next step, we fit the model to our training data with 10 epochs and we evaluated the training accuracy to see how well the model fit to the training data. We also used callbacks to understand the internal state of the model while it was training on the image data. 

For predicting labels we have a list of dictionaries. The dictionary includes the id, the prediction and the index of the correct label. Then we put the label name and the preidcted label into a json file.

### Results
For the two main training sessions -- black and white images, and edge images -- the following loss graphs were made. 
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Training_Loss_ Full_Image.png" width="450" height="320"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Training_Loss_ Edge_Image.png" width="450" height="320"/>
</p>
These graphs were plotted from the instantanious loss calculated by Keras, sampled on every 500 images. Each batch per epoch used for training was a bootstrapped sample of 10000 images from the dataset. Each "Step" in the loss graph corresponds to the beginning of one of these batches. 

For the results of our model, we split our training data into a training set as well as a tuning set. We then looked at the following
8 class labels:

          Person, Table, Tree, Building, Glasses, Boat, Insect, Dog

<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_test_full.png"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_test_edges.png"/>
</p>

Additionally, we trained a second model without the split, then tested on the 1000 non-eurocentric images provided by Kaggle, 
again against the above 8 labels.

<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_tune_full.png"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_tune_edges.png"/>
</p>

### Discussion and Conclusion
While we bootstrapped the images for both our tuning and our testing datasets, we were able to choose our method and "seed" the bootstrapping. Thanks to this, the data results from each of the 4 above tests were gathered from models trained in the exact same way. We were able to directly compare the test results with one another, being assured that each model was trained in the same way and in the same order, reducing noise from test to test.

Each label has a separate number of images associated with it. For example, one label could correspond to 20000 of the images used while another could correspond to only 10 of the images. This is a problem because by training a label with proportionally more images, we could potentially skew the predictions of our label; this would mean it would only guess what the model has seen the most. We, however, could not limit the number of images-per-label to the label with the smallest number of images, as the smallest label had magnitudes smaller images associated with it, only 2 images. To solve this, we first examined the number of images per label, and realized it was a left skewed normal distribution (See 'Other' Folder in our Github Repo). Because of this, we chose the number of images per label to be the average of the mean and median of the distribution, at just over 4000 images per label. 

While looking at the results, one can see that the percentage of eurocentric test data guessed right is much higher than the percentage of non-eurocentric. This can be explained by two separate phenomena. First, the sample size. The number of images tested on for the eurocentric testing set was about 25% of the training set. This means that the number of images corresponding to just the given labels is ~6000 to 7000. The non-eurocentric dataset had only 1000 images total, across all labels. While some images may have had multiple labels there were less than 500 images that corresponded to the 8 class labels. The second explanation is that the data is specifically non-eurocentric. This means that the images were purposely difficult to identify meaning that the expected outcome of the model was lower than the eurocentric data. The results are still valid, however, as the models were trained on the exact same base images in the same order. The testing images as well were passed as the corresponding image type, edges or black and white. 

Looking at the data shown above, we can see for both eurocentric and non-eurocentic images that preprocessing the images as edges does in fact improve the classification of objects where tonality matters. This can be seen by the increased classification of 'Person' from the eurocentric dataset from 93.2% to 99.2% and from the noneurocentric dataset from 29.2% to 29.6% for the edge images. For items where tonality does not matter, such as the 'Tree' label or the the 'Table' label, however, we can see that converting the images to purely the edges results in a higher misclassification rate. The eurocentric data changes from a classification rate of 99.6% to 97.2% for edges with label 'Tree' and the non-eurocentric data changes from a classification rate of 86.9% to 71.0%. Again, the same trend can be seen for building and boat. 

By preprocessing images as the shapes themselves, instead of the full color objects, we reduce variance caused by tonality. However, we end up removing valuable information from the image, thus leading to increased bias. For objects with high tonal variance, such as skin color, reducing the information to the edges drastically increases the accuracy of classification, as the increase of bias does not outweigh the benefits gained by smoothing the tonal noise. From this, reducing the input to purely edges is seemingly an effective way to increase the accuracy against a small number of specific euro and non-eurocentric datasets such as people. For objects with low tonal variance, or high shape variance, such as trees, houses, or boats, reducing images to just the edge outlines removes valuble information and causes underfitting: the increase in bias does not justify the smoothing of the variance. Therefore, the team concludes our proposed hypothesis is not an immediate effective way of increasing the accuracy across the majority of classes, both eurocentric and non-eurocentric. 

### To Run
1. Create a GCP account and a new project.
2. Follow the Readme in the folder PreprocessLabels
3. Follow the Readme in the folder Image PreProcessing
4. Follow the Readme in the folder MLEngineTrainers

### References
##### Academic Papers:
1. Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2017). Men also like shopping: Reducing gender bias amplification using corpus-level constraints. arXiv preprint arXiv:1707.09457.
2. Shankar, S., Halpern, Y., Breck, E., Atwood, J., Wilson, J., & Sculley, D. (2017). No Classification without Representation: Assessing Geodiversity Issues in Open Data Sets for the Developing World. arXiv preprint arXiv:1711.08536.
##### Online Sources:
- https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
- https://elitedatascience.com/keras-tutorial-deep-learning-in-python
