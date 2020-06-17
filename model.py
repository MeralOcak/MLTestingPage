#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:55:43 2020

@author: meralocak
"""


# Import dependencies
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset in a dataframe object and include only four features as mentioned
url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
include = ['Age', 'Survived'] # Only three features
df_ = df[include]

# Data Preprocessing
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Logistic Regression classifier

dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

# Save your model

joblib.dump(lr, 'lr_model.pkl')
print("Model dumped!")

# model_columns = list(x.columns)
# joblib.dump(model_columns, 'lr_model_columns.pkl')



