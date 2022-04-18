#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:42:28 2022

@author: vinceball
"""

from sklearn import tree
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import graphviz 

#%%
#load in test and training data
X = pd.read_csv('X_train.csv', delimiter = ',', thousands = ',').set_index('Unnamed: 0').sort_index()
Y = pd.read_csv('Y_train.csv', delimiter = ',', thousands = ',').set_index('Unnamed: 0').sort_index()
X_test = pd.read_csv('X_test.csv', delimiter = ',', thousands = ',').set_index('Unnamed: 0').sort_index()
Y_test = pd.read_csv('Y_test.csv', delimiter = ',', thousands = ',').set_index('Unnamed: 0').sort_index()

#%%
#remove month and year
X = X.drop(['Month', "Year"], axis=1)
X_test = X_test.drop(['Month', "Year"], axis =1)
#%%
#convert time to string
X['Time'] = pd.to_datetime(X['Time']).dt.hour.apply(str)

X_test['Time'] = pd.to_datetime(X_test['Time']).dt.hour.apply(str)

#%%
#convert strings to dummy variables

X_dumb = pd.get_dummies(data=X[['Time','Daytpe']], drop_first=False)
X_test_dumb = pd.get_dummies(data=X_test[['Time',"Daytpe"]], drop_first=False)

X.drop(['Time','Daytpe'], axis =1, inplace = True)
X_test.drop(['Time','Daytpe'], axis = 1, inplace = True)

#%%
#replace in time dummies

X = pd.concat([X, X_dumb], axis = 1)
X_test = pd.concat([X_test, X_test_dumb], axis = 1)

#Renaming for XGBoost
X.rename(columns = {'Photovoltaics[MWh]':'PV', 'Wind onshore[MWh]':'Wind Onshore', 
                    'Wind offshore[MWh]': 'Wind Offshore'}, inplace = True)
         
#%%
regressor = DecisionTreeRegressor(max_depth = 4)
clf = regressor.fit(X, Y)
#%%
plt.figure(dpi = 1000)

tree.plot_tree(clf, feature_names = X.columns)
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         