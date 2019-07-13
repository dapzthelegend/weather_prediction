# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:26:59 2019

@author: DARA
"""

from flask import Flask, render_template,url_for,request
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from model import RainfallModel
app = Flask(__name__)


#create new model
model = RainfallModel()


#load trained model
path='lib/models/Regressor.pkl'
with open(path, 'rb') as f:
    model.regressor = joblib.load(f)
    
    
#argument parsing
parser = reqparse.RequestParser()
parser.add_argument('year')


@app.route('/')
def home():
 return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
      year = request.form['year']
      year = np.reshape([[year]], (-1,1))
      rainfall= model.predict(year)
    return render_template('result.html', prediction =rainfall, year =  request.form['year'])


if __name__ == '__main__':
 app.run(debug=True)