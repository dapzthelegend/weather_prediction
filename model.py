# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:34:46 2019

@author: DARA
"""
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

 
 
 
class RainfallModel(object):
    
    def __init__(self):
        #Create your model instance
        self.regressor = RandomForestRegressor(n_estimators = 69,
                                         max_depth= 50, 
                                         bootstrap= True, 
                                         max_features="auto", 
                                         min_samples_leaf= 1, 
                                         min_samples_split=10, 
                                         random_state = 0)
   
    def split_dataset(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        self.X_train = self.X_train.reshape(-1,1)
        self.X_test = self.X_test.reshape(-1,1)
        self.y_train = self.y_train.reshape(-1,1)
        self.y_test = self.y_test.reshape(-1,1)
        self.regressor.fit(self.X_train,self.y_train)
    
    
    def predict(self, i):
        year = int(i)
        if(isinstance(year,int)):
            if(year >= 1949 and year <= 2019):
                year = self.scale(year)
                rainfall=self. regressor.predict(year)
                return "The total amount of rainfall of "+ str(int(i)) + " (January - December): " + str('{:0.2f}'.format(rainfall[0])) + "mm"
            elif(year < 1949):
                return "Enter a year greater than 1949"
            elif(year >2019):
                return "Enter a year less than 2019"
            else:
                return "Enter a valid year"
        else:
            return "Enter a valid year"
    
    def pickle_regressor(self):
        path='lib/models/Regressor.pkl'
        f = open(path, 'wb')
        f.close()
        with open(path, 'wb') as model:
            joblib.dump(self.regressor, model)
            
    def scale(self,year):
        min = 1949
        max = 1994
        if(year<1990):
            year = np.reshape([[year]], (-1,1))
        else:
            X_std = (year - 1990) / (2022 - 1990)
            year = X_std * (max - min) + min
            year = np.reshape([[year]], (-1,1))
        return year
    
   
            
            
    def getParams(self):
        parameters = {
         'max_depth': [15,20,25,30,35,40,45,50, None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [3,4,5,6,7,8],
         'min_samples_split': [6, 8, 10, 12, 14, 16],
         'n_estimators': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]}
        random_search = RandomizedSearchCV(estimator = self.regressor,
                                    param_distributions = parameters,
                                    verbose = 2,
                                    cv = 10,
                                    n_iter = 100,
                                    n_jobs =-1)
        random_search = random_search.fit(self.X_train, self.y_train)
       
    
    
    #Funtion to evaluate the model
    def evaluate(self):
        model = self.regressor
        test_features = self.X_train
        test_labels = self.y_train
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        return 'Accuracy = {:0.2f}%.'.format(accuracy)
    
    
    
   
      
       




