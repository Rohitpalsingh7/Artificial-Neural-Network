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
    
    data=pd.read_csv('~/ANN/Churn_Modelling.csv')
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
        
    def build_classifier(optimizer, units):
        classifier = Sequential()
        classifier.add(Dense(units=units, activation='relu', kernel_initializer='uniform', input_dim=11 )) 
        classifier.add(Dense(units=units, activation='relu', kernel_initializer='uniform'))
        classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
        classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return classifier
    
    model_ann = KerasClassifier(build_fn = build_classifier)
    
    parameters1 = {'batch_size': [8,16,32],
              'epochs': [100],
              'optimizer':['adam','rmsprop'],
              'units':[6,11]}
    
    parameters2 = {'batch_size': [8,16,32],
               'epochs': [300],
              'optimizer':['adam','rmsprop'],
              'units':[6,11]}
    
    grid_search1 = GridSearchCV(estimator= model_ann,
                           param_grid= parameters1,
                           scoring ='accuracy',
                           cv= 10)
    
    grid_search2 = GridSearchCV(estimator= model_ann,
                           param_grid= parameters2,
                           scoring ='accuracy',
                           cv= 10)
    


# Part 2: Tuning 
 
# Parameter tuning using concept called device parallelization which makes this process faster by computing 
# different portion of code to run in parallel on different devices...

with tf.device('/gpu:0'): 

    grid_search1 = grid_search1.fit(X_train, Y_train)
    

with tf.device('/gpu:1'): 
    
    grid_search2 = grid_search2.fit(X_train, Y_train)
    
    

with tf.device('/cpu:0'):
    
    best_para_1 =  grid_search1.best_params_
    best_accu_1 = grid_search1.best_score_
    best_para_2 =  grid_search2.best_params_
    best_accu_2 = grid_search2.best_score_
    print("\n")
    
    print("--------------------\n")
    
    if(best_accu_1 > best_accu_2) :
        print("GPU:0 tuning for epochs=100")
        print("Accuracy is ", best_accu_1)
        print("Best parameters are ", best_para_1)
        
    else :
        print("GPU:1 tuning for epochs=300")
        print("Accuracy is ", best_accu_2)
        print("Best parameters are ", best_para_2)
        
        

# Best parameter: {'batch_size': 32, 'units': 11, 'epochs': 300, 'optimizer': 'adam'} Accuracy is .86075
