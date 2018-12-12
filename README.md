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

This team project focused on the development of robust image classification machine learning models to handle pictures from geographically diverse regions (primarily non-Americas and non-European). The inspiration for this project came from a Kaggle competition called the [Inclusive Images Challenge](https://www.kaggle.com/c/inclusive-images-challenge). This is where information on the training and test image datasets can be found as well.

The team developed a traditional Convolutional Neural Network (CNN) using Tensorflow and Keras. We also had to learn and utilize the Google Cloud Platform (GCP) services; the GCP services used were Cloud Storage and Model Training due to the enormous size of the Open Images dataset. The team hypothesized that emphasizing the structure (aka. features such as edges) would allow for better generalization. The results of this project work suggest that a method that augments the dataset to support generalization is likely a better approach.

### The Data
The data used to complete this research project is part of the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html). As this is a very large dataset totaling over 500GB,for the classifyable data, our team had to use Google Cloud Platform with TensorFlow when implementing the models. We had 3 sets from this data, which we worked with: One of 200,000+ images and all 500 trainable labels, one of 37000 images, made from 8 of the most common labels, and one of 3600 images, and the same 8 labels, the final of which was used for quick tests. "Train/Test" split the data into training and testing data. "Train/Tune" was tested using the real-world test dataset, 1000 images across all 500 labels, gathered from the [Kaggle competition page](https://www.kaggle.com/c/inclusive-images-challenge/data).

Each set of images was reduced down to 256x256, then was stored as black and white and edge images, as seen below. 
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/full1.jpg"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/edge1.jpg"/>
</p>

### Image Processing
In our project, we have two sets of images. One set is the grey scale version of images and one set is only object edges in the images. We croped the edge images to the given bounding boxes and resized them to 256 x 256 after. We resized the gray scale images set to 256 x 256 as well. Our motivation for finding the edges was to minimize information to decrease bias and generalize the images. For example for facail features skin tone would not play a part. 

### Model
In order to run the model, first we perform some preprocessing steps. The first step is extracting the training and testing data from the dictionary. We split the data and dedicate 25% of the data to testind and 75% to training. We reshape the data to only have the grayscale channel. Then we convert both datasets to type float. Finally we normalize the data to get values between 0 and 1.
After the prerprocessing steps, we create the model as a sequence of layers. The first two input layers are 2D convolution
layers. Each creates a convolution kernel that convolves with the layer input to create the output. We used 32 filters for each convolution layer, a 3 by 3 convolution kernel and the relu activation function.
Our next input layer is MaxPooling2D which is used to reduce the number of parameters in the model. It uses a 2 by 2 sliding window and finds the maximum number of the 4 values.
The next input layer is Dropout which normalizes the model and that helps prevent overfitting.
We use Flatten as the next input layer to flatten the data into a 1D array.
Finally our last model is Dense which takes in the final number of outputs which corresponds to the number of classes we use for classification. For this layer we use the softmax activation function to map inputs between 0 and 1.
After adding all the input layers to our model, we compile the model with the sparse_categorical_crossentropy loss function and the adam optimizer. Based on the graph below adam performs better than the rest.
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Optimizers_graph.png" 
width="400" height="370"/>
</p>

### Results
<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Training_Loss_ Full_Image.png" width="400" height="370"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/Training_Loss_ Edge_Image.png" width="400" height="370"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_test_full.png"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_test_edges.png"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_tune_full.png"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/train_tune_edges.png"/>
</p>

### Discussion and Conclusion


### To Run
1. Create a GCP account and a new project.
2. Follow the Readme in the folder PreprocessLabels
3. Follow the Readme in the folder Image PreProcessing
4. Follow the Readme in the folder MLEngineTrainers

### References
1. Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2017). Men also like shopping: Reducing gender bias amplification using corpus-level constraints. arXiv preprint arXiv:1707.09457.
2. Shankar, S., Halpern, Y., Breck, E., Atwood, J., Wilson, J., & Sculley, D. (2017). No Classification without Representation: Assessing Geodiversity Issues in Open Data Sets for the Developing World. arXiv preprint arXiv:1711.08536.
