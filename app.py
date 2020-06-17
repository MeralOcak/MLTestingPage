#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:26:59 2020

@author: meralocak
"""


import joblib
import flask
from flask import request, jsonify, render_template
#import pandas as pd
import numpy as np

# Use pickle to load in the pre-trained model
with open('lr_model.pkl', 'rb') as f:
    model = joblib.load(f)


# with open('lr_model_columns.pkl', 'rb') as f:
#     model_columns = joblib.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return render_template('main.html', prediction_text='Prediction is {}'.format(prediction))


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug =True)
    
    


