![](UTA-DataScience-Logo.png)

# Project Title

This repository holds an attempt to teach a machine to estimate whether or not a mushroom is poisonous from a list of 22 characteristics.

## Overview

The "Mushroom Classification" dataset does not come with a particular Kaggle Challenge. Instead, parameters from outside of the Kaggle site were set to create our own challenge. I trained a model to predict whether or not a mushroom was edible based on their physical characteristics. The approach in this repository was to formulate the problem as a binary classification task with the Logistic Regression model. Originally, the Random Forest Classifier model was used, but a consistent 100% accuracy rate. A simpler model, like Logistic Regression, was favored in the end product to ensure that the code did not have a source of data leakage. After extensive data cleaning, the Logistic Regression model with several important features removed had an accuracy score of 99.7%. Other submissions on the Kaggle site had similar success, ranging from 99% to 100% accuracies.

## Summary of Workdone

### Data and Clean-Up

In total, there were 8,124 mushrooms. 22 features of each mushroom were provided that responded to their physical characteristics, ranging from color, smell, size, shape, population trends, location, etc. There was no test training set provided, so I split the data into 80% (6499 data points) for training and 20% (1,625 data points) for testing. The `stalk-root` feature had several missing data points that were distinctly labelled with "?". I cleaned these up to be properly empty (NaN) data points. I removed the `veil-type` feature because all 8,124 data points full under the same category. After getting a perfect accuracy score with different models, three features of "high importance" (`odor`, `spore-print-color`, and `gill-color`) were removed to make sure that the perfect accuracy was not from the presense of the one-hot encoded binary classification of the class.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

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







