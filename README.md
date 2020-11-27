# Rideshare Churn Predictor

This is a case study to predict churn using real-world data from a ride-sharing company. 


### The Problem Statement

What factors lead to user "churn" (leaving the app) and what are some measures 
we can implement to prevent loss? 


We would like you to use this data set to help understand **what factors are
the best predictors for retention**, and offer suggestions to help Company X. 
Therefore, your task is not only to build a
model that minimizes error, but also a model that allows you to interpret the
factors that contributed to your predictions.

### The Data

To help explore this question and to put into practice, Galvanize, the Data Science Immersive Program 
located in San Francisco, have provided a sample dataset of a cohort of users who signed up for an account 
in January 2014. The data was pulled on July 1, 2014; we consider a user retained if they were “active” 
(i.e. took a trip) in the preceding 30 days (from the day the data was pulled). In other words, a user is "active" 
if they have taken a trip since June 1, 2014. The data, churn.csv, is in the data folder. 
The data are split into train and test sets.


##### Data Manipulation & Feature Engineering

My data pipeline is outlined by the following process:

1) Sort the data by last trip date in ascending order.
2) Fill in missing ratings data with the average rating of all riders and drivers respectively.
3) Drop the remaining records that contain any sort of missing values (justified this action
        after finding just a few hundred records missing (exactly 319) from phone device data).
4) Convert last trip date from pandas object to datetime format.
5) Create a "churn" column based on standards of activity described above.
6) Create column in Boolean value if app user's device is an iPhone.
7) Convert luxury car indicator to Boolean.
8) Create dummy variables for city.





### The Model

Models:
1. Logistic Regression
2. Random Forest Classier
3. Gradient Boosting



### Conclusion






### Future 