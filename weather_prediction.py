# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:34:46 2019

@author: DARA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
#Read the file
from netCDF4 import Dataset
path = "C:\\Users\\wicho\\Downloads\\New folder\\data.nc"
dataset = Dataset(path)


#Get the data keys['data', 'time', 'lat', 'lon']
d = dataset.variables['data'][:,:,:]
time = dataset.variables['time'][:]
long = dataset.variables['lon'][:]
lat = dataset.variables['lat'][:]
data = []

#Melthing the 3D array to a 2D array
def melt():
    count = 0    
    for year in range (46):
        
        if((year+1)%4 ==0 & year != 0):
            count = count + 367
            data.append([1949+year,d[count-367:count, : , :].sum() ])
        else:
            count = count + 366
            data.append([1949+year,d[count-366:count, : , :].sum() ])
            
melt()
            
#Reshape       
data = np.reshape(data, (46,2))

#Split the dataset
x = data[:, 0]
y = data[:, -1]

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#Reshape
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
x = x.reshape(-1,1)
y = y.reshape(-1,1)


#Creating a Random Forest Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 69,
                                     max_depth= 50, 
                                     bootstrap= True, 
                                     max_features="auto", 
                                     min_samples_leaf= 5, 
                                     min_samples_split=12, 
                                     random_state = 0)
regressor.fit(X_train,y_train)


#Predit using RF
y_pred = regressor.predict(X_test)
year = np.reshape([[2019]], (-1,1))
this_year = regressor.predict(year)


#Appying Grid Search to find the best model and best parameters
from sklearn.model_selection import RandomizedSearchCV
def getParams():
    parameters = {
     'max_depth': [15,20,25,30,35,40,45,50, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [3,4,5,6,7,8],
     'min_samples_split': [6, 8, 10, 12, 14, 16],
     'n_estimators': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]}
    random_search = RandomizedSearchCV(estimator = regressor,
                                param_distributions = parameters,
                                verbose = 2,
                                cv = 10,
                                n_iter = 100,
                                n_jobs =-1)
    random_search = random_search.fit(X_train, y_train)
    best_accuracy = random_search.best_score_
    best_parameters = random_search.best_params_


#Funtion to evaluate the model
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy


def predictRainfall():
    y = input("Enter the year in question: ")
    year = int(y)
    if(isinstance(year,int)):
        year = np.reshape([[year]], (-1,1))
        rainfall= regressor.predict(year)
        print ("The total amount of rainfall of "+ str(year[0][0]) + " (January - December): " + str('{:0.2f}'.format(rainfall[0])) + "mm")
        evaluate(regressor, X_train, y_train)
        
    else:
        print("Enter a valid year")
        
#predict the rainfall
predictRainfall()
        
        
#
    




