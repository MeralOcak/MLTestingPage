#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:26:59 2020

@author: meralocak
"""


import joblib
import flask
from flask import request, jsonify, render_template
import numpy as np

# Use pickle to load in the pre-trained model
with open('lr_model.pkl', 'rb') as f:
    model = joblib.load(f)

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



# Set up the main route
# @app.route('/', methods=['GET', 'POST'])
# def main():
#     if flask.request.method == 'GET':
#         # Just render the initial form, to get input
#         return(flask.render_template('main.html'))
    
#     if flask.request.method == 'POST':
#         # Extract the input
#         Age = flask.request.form['Age']
    

#         # Make DataFrame for model
#         input_variables = pd.DataFrame([Age],columns=['Age'],dtype=float, index=['input'])

#         # Get the model's prediction
#         prediction = model.predict(input_variables)[0]
    
#         # Render the form again, but add in the prediction and remind user
#         # of the values they input before
#         return flask.render_template('main.html',
#                                      original_input={'Age':Age},
#                                      result=prediction,
#                                      )

if __name__ == '__main__':
    app.run(debug =True)
    
    


