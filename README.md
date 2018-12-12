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
The data used to complete this research project is part of the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html). As this is a very large dataset totaling over 500GB,for the classifyable data, our team had to use Google Cloud Platform with TensorFlow when implementing the models. Due to the technical hurdles and project scope, we implemented just a portion of the image dataset. The data was split into training and testing data sets for the dataset called "Train/Test." The real-world test dataset was gathered from the [Kaggle competition page](https://www.kaggle.com/c/inclusive-images-challenge/data), which we combined with our training set to create the "Train/Tune" dataset.

<p align="center">
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/edge1.jpg"/>
<img src="https://raw.githubusercontent.com/cabincabin/MLRobustClassifier/master/img/full1.jpg"/>
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
