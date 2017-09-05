# Artificial Neural Network

# Business Problem : To determine whether customer will leave the bank or not (i.e churn rates) 
# This solution can help bank to decide whether loan or credit card should be approved or not etc for a particular customer
# So using ANN classifier we can get sorted list of all customers with highest probability and we can target them.

# Part 1 - Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('Churn_Modelling.csv')
X= data.iloc[:, 3:13].values
Y= data.iloc[:, 13].values

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
X=X[:,1:]

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


# Part 2 - Designing ANN Architecture 

# Sequential model is used to initialize and Dense is used to add layer
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11 )) 
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# Part 3 - Making prediction on new single observation 

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

predict_new= classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1,1, 50000]])))
predict_new=(predict_new>0.5)


# Part 4 : Evaluating, Improving and Tuning the ANN 

# Each time Model will run then it will give different accuracy on training and test set 
# so solution is k-fold cross validation.

# Evaluating 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential 
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11 )) 
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
model_ann = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100) 
accuracies = cross_val_score(estimator = model_ann, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
mean_accuracy = accuracies.mean()
variance_acc = accuracies.std()

# Tuning 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', kernel_i   nitializer='uniform', input_dim=11 )) 
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
model_ann = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [15,25,32],
              'epochs': [200, 500],
              'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator= model_ann,
                           param_grid= parameters,
                           scoring ='accuracy',
                           cv= 10)
grid_search = grid_search.fit(X_train, Y_train)
best_para =  grid_search.best_params_
best_accu = grid_search.best_score_

