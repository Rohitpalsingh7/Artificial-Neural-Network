#!/bin/env python3


#SBATCH -N 1


#SBATCH -n 2


#SBATCH --mem=26G


#SBATCH -p short


#SBATCH -C K40


#SBATCH -o tf_test.out


#SBATCH -t 24:00:00


#SBATCH --gres=gpu:2

# Artificial Neural Network

# Business Problem : To determine whether customer will leave the bank or not (i.e churn rates) 
# This solution can help bank to decide whether loan or credit card etc should be approved or not for a particular customer
# So using ANN classifier we can get sorted list of all customers with highest probability and we can target them.

# Part 1 - Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; 
import keras.utils
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #Used for regulaization to reduce overfitting if any
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

with tf.device('/cpu:0'): 

    data=pd.read_csv('Churn_Modelling.csv')
    X= data.iloc[:, 3:13].values # Here X is a matrix not vector
    Y= data.iloc[:, 13].values # Y is a vector
    
    # Encoding categorical data and create a dummy variable 
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    le_1=LabelEncoder()
    X[:,1]=le_1.fit_transform(X[:,1])
    list(le_1.classes_)
    le_2=LabelEncoder()
    X[:,2]=le_2.fit_transform(X[:,2])
    list(le_2.classes_)
    hot_encoder=OneHotEncoder(categorical_features=[1])
    X=hot_encoder.fit_transform(X).toarray()
    X=X[:,1:] # Now X is an 2D array 
    
    # Split the data into training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=0)
    
    # Perform Standardization 
    # centric and scaling of a feature var is determined by parameters ..
    from sklearn.preprocessing import StandardScaler 
    sc=StandardScaler()
    X_train= sc.fit_transform(X_train)
    sc.mean_
    X_test = sc.transform(X_test)

    def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(units=11, activation='relu', kernel_initializer='uniform', input_dim=11 )) 
        classifier.add(Dense(units=11, activation='relu', kernel_initializer='uniform'))
        classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return classifier
    
    model_ann = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 300)
    variance_acc = 0
    
# Part 2 : Evaluating and Improving 

# Each time Model will run then it will give different accuracy on training and test set so solution is k-fold cross validation.
# Evaluating with best hyperparameters 

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

with tf.device('/gpu:0'):
    
    accuracies = cross_val_score(estimator = model_ann, X = X_train, y = Y_train, cv = 10)
    mean_accuracy = accuracies.mean()
    variance_acc = accuracies.std()
        
    print("\n\nMean training accuracy of 10 cross-fold validation is :", mean_accuracy)
    print("Variance of 10 cross-fold validation is :", variance_acc)
    

# Part 3 - Designing ANN Architecture with best hyperparameters and if variance is high then use dropout for overfitting

with tf.device('/gpu:1'):
    
    if(variance_acc > 0.01) :    
    
        print("variance is high so building model using dropout.")
        classifier = Sequential()
        classifier.add(Dense(units=11, activation='relu', kernel_initializer='uniform', input_dim=11 )) 
        classifier.add(Dropout(rate=0.1))
        classifier.add(Dense(units=11, activation='relu', kernel_initializer='uniform'))
        classifier.add(Dropout(rate=0.1))
        classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(X_train, Y_train, batch_size=32, epochs=300)
        Y_pred = classifier.predict(X_test)
        
    else : 
        
        model_ann.fit(X_train, Y_train)
        Y_pred = model_ann.predict(X_test)
        
    Y_pred = (Y_pred > 0.5)
    cm = confusion_matrix(Y_test, Y_pred)
    print("----------------------------------------\n")
    print("Confusion Matrix based on threashold value = 0.5 :\n", cm)
        
        
        
        
            
#----------------------------------------------------------------------------------------------------#
        
        
# Part 4 - Making prediction on new single observation 

""" To check if customer will leave the company or not :
Country : France
Credit Score : 600
GEnder : Male
Age : 40 
Tenure : 3 
Balance : 60000
#Products : 2 
Has Credit Card : Yes 
Is Active Member : Yes 
Estimated Salary : 50000 """

#predict_new= classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1,1, 50000]])))
#predict_new=(predict_new>0.5)

# Sequential model is used to initialize neural network as sequence of layers and Dense is used to add layer



