# Artificial-Neural-Network
Artificial neural network model to predict whether customer will leave the bank or not. 

ANN model is implemented using keras library in python and it used tensorflow backend. 

## Configuration of Clusters 

-- ANN tuning and model evaluation is done on High performance computing clusters (HPC). Configuration of the cluster is K40 with 2 GPUs. It has time limitation of 48 hours and terminate the job exceeding the defined time limit. 

-- Clusters deployed over Red Hat Enterprise Linux 7.2

## Steps used for predicting churn rates of customers :

1. Performed hyperparameters tuning on two GPUs in succession to prevent load on one GPU.
2. File 'Tuning.py' gives best hyperparameters based on accuracy metric. Used gridsearch with 10 cross fold validation to get the best hyperparameters. 
3. Using the above best hyperparameters, ANN (ANN_model.py) is evaluated by 10 cross fold validation and mean accuracy and variance of 10 fold is computed. 
4. VArinace indicates whether model is overfitted or not. I used threashold value of 0.01 valriance and if computed variance 
crosses the threashold value then regularization (dropout) is used to deactivate few neurons in hidden layer.
5. Predicted churn rates on test set. 

## Best Hyperparameters :

1. {'batch_size': 32, 'units': 11, 'epochs': 300, 'optimizer': 'adam'}
2. Adam is the stochastic gradient descend optimization algorithm. 

## Results :

1. Training accuracy with 10 CV: 85.31 %
2. VAriance of 10 CV: 0.012
3. Final predictions are probabilities so with a threashold value of 0.5, churn rates are predicted.
4. Test accuracy : 84.30 %
