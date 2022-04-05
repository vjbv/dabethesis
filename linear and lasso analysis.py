#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:27:16 2022

@author: vinceball
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

import matplotlib.pyplot as plt
import seaborn as sns


#%%
#load data

unstandardized_data = pd.read_csv("full_dataset.csv", delimiter=",", thousands=',').dropna()

#%%
#one-hot encoding day of the week
one_hot_daytype = pd.get_dummies(unstandardized_data.Daytype)

#%%
#add special variables 

#photovoltaics/wind
unstandardized_data['P/W']= unstandardized_data['Photovoltaics[MWh]']/(unstandardized_data['Wind offshore[MWh]']+unstandardized_data['Wind onshore[MWh]'])

#renewable/total
unstandardized_data['Renewable/Resid'] = (unstandardized_data['Photovoltaics[MWh]']+unstandardized_data['Wind offshore[MWh]']+unstandardized_data['Wind onshore[MWh]']) / unstandardized_data['Residual load[MWh]']
#%%
#list the variables from the imported csv that you want

floating_list = ['Spot Price', 'Total (grid load)[MWh]', 'Hydro pumped storage[MWh]_x', 'Biomass[MWh]', 'Hydropower[MWh]',
                 'Wind offshore[MWh]', 'Wind onshore[MWh]','Photovoltaics[MWh]', 'Other renewable[MWh]', 'Nuclear[MWh]', "Lignite[MWh]",
                 'Hard coal[MWh]', "Fossil gas[MWh]", 'Hydro pumped storage[MWh]_y', 'Other conventional[MWh]','P/W','Renewable/Resid']
#%%
#standarized specific columns
standardized_data = pd.DataFrame()
for i in floating_list:
    standardized_data[i] = (unstandardized_data[i] - unstandardized_data[i].mean()) / unstandardized_data[i].std()
#%%
#get the variables you want to regress on from the standardized dataframe from your floating_list
standardized_data = standardized_data[np.intersect1d(standardized_data.columns, floating_list)]
X = standardized_data.drop('Spot Price', axis = 1)

#take unstandardized ratio spot monthly as target
Y = unstandardized_data['Ratio Spot Monthly']


#%%
#creating interaction variables for all included variables
def create_interactions(df):
    df_int = df.copy()
    for i in range(0, len(df.columns)-1):
        for j in range(i+1, len(df.columns)):
            name = str(df.columns[i]) + ' * ' + str(df.columns[j])
            df_int.loc[:, name] = df[str(df.columns[i])] * df[str(df.columns[j])]
    return df_int

#%%
#create polynomials for all variables
poly_vars = pd.DataFrame()
for i in floating_list:
    poly_vars[i] = standardized_data[i] * standardized_data[i]
    poly_vars.rename(columns ={i : i + '^2'}, inplace = True)
    
poly_vars=poly_vars.drop('Spot Price^2', axis = 1)

#%%
#create interaction dataframe

interactions_df = create_interactions(X)

#add one_hot and polynomial transformations

lasso_data = pd.concat([interactions_df, one_hot_daytype, poly_vars, unstandardized_data['Covid']], axis = 1)


#%%
#create linear model
linear_model = LinearRegression()
linear_model.fit(lasso_data, Y)
print(linear_model.score(lasso_data, Y))
#r_sq = model.score(x, y)
#model.fit(x, y)
#model = LinearRegression()


#%%
#create Lasso Model for first analysis

# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
model.fit(lasso_data, Y)

#get variables
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(lasso_data, Y)

#%%
#get coefficients
lasso_coefficients = list(zip(lasso_best.coef_, interactions_df))
lasso_coefficients = pd.DataFrame(lasso_coefficients)
Non_zero_lasso = lasso_coefficients[lasso_coefficients.iloc[:,0] != 0]

#get score
print('R squared training set', round(lasso_best.score(lasso_data, Y)*100, 2))

#%%
#plot models
plt.style.use('dark_background')
fig, ax = plt.subplots(2,1, sharey = True)

ax[1].plot(lasso_data.index, lasso_best.predict(lasso_data), color = 'r', lw = .3)
ax[1].scatter(lasso_data.index, Y,alpha = .1, marker = 'o')
ax[1].set_title('Lasso Model Predictions')
ax[1].set_ylabel('Price/Monthly Average')

ax[0].plot(lasso_data.index, linear_model.predict(lasso_data), color = 'r', lw = .3)
ax[0].scatter(lasso_data.index, Y, alpha = .1, marker = 'o')
ax[0].set_title('Linear Model Predictions (R^2 .78)')
ax[0].set_ylabel('Price/Monthly Average')

plt.show()

#%%
#build and plot hist of resids
residuals_linear = (linear_model.predict(lasso_data) - Y).reset_index()
residuals_lasso = (lasso_best.predict(lasso_data) - Y).reset_index()

fig, axs = plt.subplots(2,1, constrained_layout=True)

sns.histplot(x=residuals_linear['Ratio Spot Monthly'], ax = axs[0])
axs[0].set_title('Linear Residuals')
axs[0].set_xlabel('Residuals')
axs[0].set_ylabel('Count')

sns.histplot(x = residuals_lasso['Ratio Spot Monthly'], ax = axs[1])
sns.histplot()
axs[1].set_title('Lasso Residuals')
axs[1].set_xlabel('Residuals')
axs[1].set_ylabel('Count')


plt.legend()
plt.show()



#axs[0].set_title('subplot 1')
#axs[0].set_xlabel('distance (m)')
#axs[0].set_ylabel('Damped oscillation')


