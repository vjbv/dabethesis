#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:03:55 2022

@author: vinceball
"""
'''
Try to load data in each section corresponding to each graph using unique names. 
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%

sns.set()

#%%

data1 = pd.read_csv("Energy Data Thiqq.csv", delimiter=",", thousands=',').dropna()

filter_list = ['Date_Time of day','Month', 'Time', 'Year', 'Date', 'Spot Price']

data1 = data1[filter_list]


#%%
def weekday_definition(orig_weekday_type):
    if orig_weekday_type == 0:
        return 'Monday'
    elif orig_weekday_type ==1:
        return 'Tuesday'
    elif orig_weekday_type == 2:
        return 'Wednesday'
    elif orig_weekday_type == 3:
        return 'Thursday'
    elif orig_weekday_type == 4:
        return 'Friday'
    elif orig_weekday_type == 5:
        return 'Saturday'
    elif orig_weekday_type == 6:
        return 'Sunday'
    
#%%
data1['Day of Week'] = pd.to_datetime(data1['Date_Time of day']).dt.day_of_week

#%%

data1['Daytype'] = data1['Day of Week'].apply(weekday_definition)

#%%

data2 = data1.groupby(["Daytype","Time"])["Spot Price"].mean()
data2 = data2.reset_index()

#%%
Daytype_sum = data2.groupby(['Daytype'])['Spot Price'].sum()
Daytype_sum = Daytype_sum.reset_index()

#%%
Daytype_avg = data2.groupby(['Daytype'])['Spot Price'].mean()
Daytype_avg = Daytype_sum.reset_index()
Daytype_avg['Weekly Sum'] = Daytype_avg['Spot Price'] / Daytype_avg['Spot Price'].sum()

#%%

data2 = pd.merge(
    data2,
    Daytype_sum,
    how='left',
    on = 'Daytype'
    )

#%%
data3 = pd.DataFrame()

data3['Hour'] = data2['Time']
data3['Daytype'] = data2['Daytype']
data3['hourly weight'] = data2['Spot Price_x']/data2['Spot Price_y']

#%%
fig, ax = plt.subplots()
fig.set_size_inches(8, 4)
sns.lineplot(data = data3, x='Hour', y = 'hourly weight', hue = 'Daytype')
ax.tick_params(rotation=45)
plt.show()

#%%
'''
here are the plots for variable analysis. PV, Wind, and Renew/Resid
'''

#%%
plotting_data = pd.read_csv("Energy Data Thiqq.csv", delimiter=",", thousands=',').dropna()
plotting_data['Renewable/Resid'] = (plotting_data['Photovoltaics[MWh]']+plotting_data['Wind offshore[MWh]']+plotting_data['Wind onshore[MWh]']) / plotting_data['Residual load[MWh]']
#rename variables for better plotting
plotting_data.rename(columns = {'Ratio Spot Monthly':'Hourly Price / Monthly Avg Price'}, inplace =True)
#%%
#histogram of monthly ratios

sns.histplot(plotting_data['Hourly Price / Monthly Avg Price'], stat='proportion')
plt.show()

#%%
#wind and photovoltaics vs spot price ratio

fig, axes = plt.subplots()
fig.set_size_inches(10, 4)

sns.scatterplot(data = plotting_data,
                y='Renewable/Resid', x = 'Hourly Price / Monthly Avg Price', alpha = .5, hue = "Daytype")
axes.set_ylabel("Solar and Wind / Non-renewables")
plt.title("Behavior of Share of Renewables with Price") 
plt.show()

#%%
#plot photo and wind
fig, axes = plt.subplots(1,2)
fig.set_size_inches(10, 4)

sns.scatterplot(data = plotting_data,
                y='Photovoltaics[MWh]', 
                x = 'Hourly Price / Monthly Avg Price', 
                alpha = .5, 
                hue = "Daytype", 
                ax=axes[0],
                legend = False)


sns.scatterplot(data = plotting_data,
                y='Wind offshore[MWh]',
                x = 'Hourly Price / Monthly Avg Price', 
                alpha = .5, 
                hue = "Daytype", 
                ax=axes[1])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()

