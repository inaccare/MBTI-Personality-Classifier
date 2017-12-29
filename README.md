# MBTI-Personality-Classifier

Please see "A Language Model of the Myers-Briggs Personality Type Index" pdf for a full description of the project with results.
For a short summary, please read the below.

For this project, we attempted to classify people into their MBTI personality type based off of their social media posts.
The data we used for training and testing came from a Kaggle dataset from social media posts at personalitycafe.com.
In order to do this, we utilized linear classifiers with a variety of different feature vectors (i.e. bucket of words, sentiment, etc.)
whose performance we compared to each other.  We also built a neural network using a multilayer perceptron classifier with
a limited-memory BFGS logistic regression schema.
