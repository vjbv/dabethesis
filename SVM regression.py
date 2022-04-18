#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:59:49 2022

@author: vinceball
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_squared_error, make_scorer
import math

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import time

#%%
#load in test and training data
full_data = pd.read_csv('data_apr_8.csv', delimiter=",", thousands=',').dropna()
X = pd.read_csv('X_train.csv', delimiter = ',', thousands = ',').set_index('Unnamed: 0').sort_index()
Y = pd.read_csv('Y_train.csv', delimiter = ',', thousands = ',').set_index('Unnamed: 0').sort_index()
X_test = pd.read_csv('X_test.csv', delimiter = ',', thousands = ',').set_index('Unnamed: 0').sort_index()
Y_test = pd.read_csv('Y_test.csv', delimiter = ',', thousands = ',').set_index('Unnamed: 0').sort_index()

#%%
x = np.linspace(0, 2*math.pi, 49)

#%%
t = np.linspace(0,23,24)
sin_of_x = pd.DataFrame()
sin_of_x['sin'] = np.sin(x)

#%%


sin_of_x.drop(sin_of_x.tail(25).index,inplace = True)

#t = pd.to_datetime(t, format='%H')[:24]
sin_of_x['Time']= full_data['Time'][0:24]
#sin_of_x['Time'] = sin_of_x['Time'].dt.hour


#sin_of_x.drop(sin_of_x.tail(1).index,inplace = True)

#%%


X = pd.merge(X,
             sin_of_x,
             how = 'left',
             on = 'Time')

X_test = pd.merge(X_test,
             sin_of_x,
             how = 'left',
             on = 'Time')
#%%
#remove month and year
X = X.drop(['Time','Month', "Year"], axis=1)
X_test = X_test.drop(['Time', 'Month', "Year"], axis =1)



#%%
#convert strings to dummy variables

X_dumb = pd.get_dummies(data=X['Daytype'], drop_first=False).to_numpy()
X_time = X[['sin_y']].to_numpy()
X_test_dumb = pd.get_dummies(data=X_test['Daytype'], drop_first=False).to_numpy()
X_test_time = X_test[['sin_y']].to_numpy()

#%%
#remove "Time" from X data
X.drop(['sin_x','cos', 'Daytype'], axis = 1, inplace = True)
X_test.drop(['sin_x','cos','Daytype'], axis = 1, inplace = True)


#%%
#scale input variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_Y = StandardScaler()

X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y).ravel()
#Y=Y.to_numpy()

X_test = sc_X.fit_transform(X_test)
Y_test = sc_Y.fit_transform(Y_test).ravel()
#Y_test = Y_test.to_numpy()


#%%

#concat in dummy variables
X = np.concatenate([X, X_dumb, X_time], axis = 1)
X_test = np.concatenate([X_test, X_test_dumb, X_test_time], axis = 1)

#np.concatenate((a, b), axis=0)
#Dataframe.to_numpy()

#%%


regressor = SVR(C = .5, gamma = 'auto', kernel = 'rbf')
regressor.fit(X, Y)

#%%

Y_hat = regressor.predict(X_test)
#%%
SVM_R2 = r2_score(Y_test, Y_hat) 
SVM_MSE = mean_squared_error(Y_test, Y_hat)

#%%
'''
#plot regression
plt.style.use('dark_background')
fig, ax = plt.subplots()

ax.plot(X_test_dumb.index, regressor.predict(X_test), color = 'r')
ax.scatter(X_test_dumb.index, Y_test, alpha = .1, marker = 'o' )
ax.set_title('SVM Model Predictions')
ax.set_ylabel('Hour/Daily Sum')

plt.show()

#%%
X_grid = np.arange(min(X), max(X), 0.01) #this step required because data is feature scaled.
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''
#%%
start = time.time()
mse = make_scorer(mean_squared_error)


param_grid = {'C': [ 0, .1],
              'gamma': ['auto'],
              'kernel': ['rbf']}

CVSVR = GridSearchCV(SVR(),
                    param_grid,
                    refit=True,
                    verbose=1,
                    scoring= mse)
CVSVR.fit(X, Y)

print(CVSVR.best_params_)

Y_hat_CV = CVSVR.predict(X_test)
#%%

SVM_R2_CV = r2_score(Y_test, Y_hat) 
SVM_MSE_CV = mean_squared_error(Y_test, Y_hat_CV)

end = time.time()

print((end - start)/60)




