import pandas as pd
import numpy as np
import scipy as sc
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split

# Importing data
df0 = pd.read_csv('./data/churn_train.csv')
df = df0.copy()

# Data Manipulation/Feature Engineering
def pipeline(df):
    df = df.sort_values(by = ['last_trip_date'])
    df['avg_rating_of_driver'].fillna(df['avg_rating_of_driver'].mean(), inplace = True)
    df['avg_rating_by_driver'].fillna(df['avg_rating_by_driver'].mean(), inplace = True)
    df.dropna(axis = 0, inplace = True)
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['churn'] = df['last_trip_date'] < '2014-06-01'
    df['churn'] = df['churn'] * 1
    df['phone'] = df['phone'] == 'iPhone'
    df['phone'] = df['phone'] * 1
    df['luxury_car_user'] = df['luxury_car_user'] * 1
    df = pd.get_dummies(df, columns = ['city'], drop_first = False)
    return 

X = df.drop(columns = ['signup_date','last_trip_date','churn'])
y = df['churn']

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 11)

# Metrics function
def print_metrics(y_test, y_pred):
    print("Accuracy Score: ", accuracy_score(y_test, y_pred))
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred, labels = [1,0]))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Precision Score: ", precision_score(y_test, y_pred))
    print("Recall Score: ", recall_score(y_test, y_pred))
    print(" F1 Ccore: ", f1_score(y_test, y_pred, average = 'weighted'))
    return

# Predicted targets
def predict_model(X_train, y_train, X_test, y_test, model, trees = 100):
    if model == LogisticRegression:
        model = model()
    else:
        model = model(n_estimators = trees)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:,1]
    return y_pred, y_pred_probs

# Function for Plotting ROC-Curves
def visualize(y_test,y_pred_probs1, y_pred_probs2, y_pred_probs3, model1, model2, model3):
    fpr1, tpr1, _ = roc_curve(y_test, y_pred_probs1)
    auc1 = roc_auc_score(y_test, y_pred_probs1)
    fpr2, tpr2, _ = roc_curve(y_test, y_pred_probs2)
    auc2 = roc_auc_score(y_test, y_pred_probs2)
    fpr3, tpr3, _ = roc_curve(y_test, y_pred_probs3)
    auc3 = roc_auc_score(y_test, y_pred_probs3) 
    plt.figure(1,figsize=(12,8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr1, tpr1, label=f'{model1} AUC={round(auc1,3)}')
    plt.plot(fpr2, tpr2, label=f'{model2} AUC={round(auc2,3)}')
    plt.plot(fpr3, tpr3, label=f'{model3} AUC={round(auc3,3)}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    return 

# Running Models
y_pred1, y_pred_probs1 = predict_model(X_train, y_train, X_test, y_test, LogisticRegression)
y_pred2, y_pred_probs2 = predict_model(X_train, y_train, X_test, y_test, RandomForestClassifier)
y_pred3, y_pred_probs3 = predict_model(X_train, y_train, X_test, y_test, GradientBoostingClassifier)

# ROC Curve
visualize(y_test,y_pred_probs1, y_pred_probs2,y_pred_probs3, 'LR', 'RF','GB')