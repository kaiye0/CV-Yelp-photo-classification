# CV-Yelp-photo-classification

Finding a good restaurant for some favorite food is always one of the best options to relax after one day’s work or study. Yelp app is one of the most widely-known apps to search for foods and restaurants. This project aims to tag restaurants with different labels based on the photos submitted by the users (that is, data set given by Yelp) using machine learning. This kind of automatic labeling would help users find their target restaurants faster and more efficiently.
Ref:https://www.kaggle.com/c/yelp-restaurant-photo-classification 

Photos of foods in different restaurants are given by Yelp in both training and test sets. The problem is to build a machine learning model to clarify these photo-related restaurants with different tags.
Generally the project can be divided into two steps:
1. Extract features from the given phots of the training set and choose useful features.
2. Build a machine learning model based on those features and train the model to achieve better results.
 
Features extracted from the photos are implemented by histogram of oriented gradient (HOG) algorithm  technique compared with connvolutional neural network(CNN) trained on ImageNet.  And the model is built using logistic regression(LR) and support vector machine(SVM)based the feature extracted. 
