# -*- coding: utf-8 -*-
#import the libaries
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from model import RainfallModel




def build_model():
    
    global d
    model = RainfallModel()
    path = 'data.nc'
    dataset = Dataset(path)
    d = dataset.variables['data'][:,:,:]
    
    
    #melt the 3D array to 2D
    melt()
    #spit the dataset
    X = data[:, 0]
    y = data[:, -1]
    model.split_dataset(X,y)
    model.pickle_regressor()

    
    
def melt():
    global data
    data = []
    count = 0    
    for year in range (46):
        if((year+1)%4 ==0 & year != 0):
            count = count + 367
            data.append([1949+year,d[count-367:count, : , :].sum() ])
            
        else:
            count = count + 366
            data.append([1949+year,d[count-366:count, : , :].sum() ])
    data = np.reshape(data, (46,2))



if __name__ == "__main__":
    build_model()


        
        
        





































































     