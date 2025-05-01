![](UTA-DataScience-Logo.png)

# Project Title

This repository holds an attempt to teach a machine to estimate whether or not a mushroom is poisonous from a list of 22 characteristics.

## Overview

The "Mushroom Classification" dataset does not come with a particular Kaggle Challenge. Instead, parameters from outside of the Kaggle site were set to create our own challenge. I trained a model to predict whether or not a mushroom was edible based on their physical characteristics. The approach in this repository was to formulate the problem as a binary classification task with the Logistic Regression model. Originally, the Random Forest Classifier model was used, but a consistent 100% accuracy rate. A simpler model, like Logistic Regression, was favored in the end product to ensure that the code did not have a source of data leakage. After extensive data cleaning, the Logistic Regression model with several important features removed had an accuracy score of 99.7%. Other submissions on the Kaggle site had similar success, ranging from 99% to 100% accuracies.

## Summary of Workdone

### Data and Clean-Up

In total, there were 8,124 mushrooms. 22 features of each mushroom were provided that responded to their physical characteristics, ranging from color, smell, size, shape, population trends, location, etc. There was no test training set provided, so I split the data into 80% (6499 data points) for training and 20% (1,625 data points) for testing. The `stalk-root` feature had several missing data points that were distinctly labelled with "?". I cleaned these up to be properly empty (NaN) data points. I removed the `veil-type` feature because all 8,124 data points full under the same category. After getting a perfect accuracy score with different models, three features of "high importance" (`odor`, `spore-print-color`, and `gill-color`) were removed to make sure that the perfect accuracy was not from the presense of the one-hot encoded binary classification of the class.

#### Data Visualization

This "Importance Chart" ranks each feature on how much they differ between edible and poisonous mushrooms based on their value distributions.
![](figure1.png)

For instance, `odor` was a very strong feature for differentiating between edible and poisonous mushrooms. Each category in every feature was almost exclusively full of only edible or only poisonous mushrooms.
![](figure2.png) 

On the other hand, `cap-shape` has several categories whose value distributions were very similar, such as having a convex, flat, or knobbed cap shape. This feature will be less helpful than others for my machine model to differentiate between classees.

### Problem Formulation

The input is a set of features corresponding to physical trait. The output is a binary label where 0 represents edible, and 1 represents poisonous. Originally, Random Forest was the model of choice, but the decision tree yielded a 100% accuracy rate and raised a concern of a faulty dataset. The Logistic Regression was favored due to being a simpler model and would provide a better metric to see if data cleanup was unsuccessful or if there was data leakage. I used scikit-learn's default solver, lbfgs, with an arbitrary seed value, 42, for reproducibility. No hyperparameter tuning was performed in this phase.

### Training

Training was performed in a jupyter notebook via Google Colaboratory with several imported packages: pandas, seaborn, matplotlib, and several sklearn packages. This project was completed on a standard Windows device. The dataset was small, so no GPU acceleration or trimming was required. Training only took a few seconds because Logistic Regression is a relatively simple model and the dataset was pretty small. The use of Logistic Regression trains the model without epochs, so there are no training curves to provide. Training stopped automatically because several test runs got 100% accuracy ratings. I ran into several difficulties pertaining the suspiciously high accuracy of my machine. I eliminated several factors to make sure that everything was coded properly. Firstly, I simplified the model I used by swapping from a decision tree to Logistic Regression. This still resulted in a perfect accuracy rating, so I continued removing highly important features, like `odor`, `spore-print-color`, and `gill-color`, until I got an accuracy rate of 99.7%. I decided to not continue removing features, as this imperfect score guarantees that I did not make an error with my one-hot encoded class.

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







