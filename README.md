# Supervised Machine Learning Homework
# Predicting Credit Risk

In this challenge, I built a machine learning model to predict whether a loan from LendingClub will become high risk or not.

Specifically, I compared the <b>Logistic Regression</b> model and <b>Random Forest Classifier</b>.

## Pre-processing

The training data used is a csv file of 2019 loans used to predict the credit risk of loans from the first quarter of the next year (2020).

I separated the target from the features and converted the features that are categorical to numeric using pd.get_dummies() in the training and testing datasets. The training dataset ended up having one more column than the test data after the dummies function and I had to use code to add the missing column(s) to prevent an error in the model.

## Fitting the models

First, I needed to predict which model would preform better. My hypothesis was the <b>Random Forest Classifier</b>. This is because of its randomness in using different decision trees made up of different random features. The feature dataset has more than 80 features and my thought is the Logistic Regression model might overfit. The Random Forest Classifier might do a better job in using different scenarios to determine high or low risk.

I first created the models and fit the data without scaling the data so that I can compare and learn the importance of scaling.
