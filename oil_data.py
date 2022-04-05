# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:43:31 2022

@author: Thomas
"""

#%% Oil Prices
import pandas as pd
import matplotlib.pyplot as plt
oil = pd.read_csv("Oil_Prices.csv", delimiter=",", thousands=',').dropna()
# data variable from test/train code
oil_merge =pd.merge(
    data,
    oil,
    how="inner",
    on=['Date']
)
plt.plot(oil_merge[['Oil_Price', 'Spot Price']])



#%%

drop_vars = ["Date_Time of day", "Day of Week", "Year",
               "Time", "Ratio Spot Monthly", "Ratio Spot Daily",
               "Daytype", "Month", "Date"]


ave_spot_price = ["Ratio Spot Monthly", "Ratio Spot Daily"]
#full_stand_data = full_stand_data.drop(full_stand_data.index[[5830,]]) drop na before trick

# Only standardized variables
stand_oil = oil_merge.drop(drop_vars, axis = 1)

# standardized with ave prices
stand_oil =pd.concat([stand_oil, oil_merge[ave_spot_price]], axis=1)
def standardize(series):
    return (series - series.mean()) / series.std()
for col in stand_oil.columns:
    stand_oil[col] = standardize(stand_oil[col])
    
#%% Plots
    
plt.plot(stand_oil[[ 'Ratio Spot Monthly', 'Oil_Price']])

plt.plot(stand_oil[[ 'Spot Price', 'Oil_Price']])
plt.ylabel(' Standardized Price')
plt.xlabel('time')
plt.legend(["Energy Spot price", "Oil Spot Price"], loc ="upper left")


