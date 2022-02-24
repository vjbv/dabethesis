#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:23:23 2022

@author: vinceball
"""
#%%
"""
Import packages for ETL process
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
"""
reading all relevant CSV
"""
ActualConsumption1 = pd.read_csv("Actual_consumption_18_20.csv", delimiter = ';')
ActualConsumption2 = pd.read_csv("Actual_consumption_20_22.csv", delimiter = ';')
ForecastedConsumption1 = pd.read_csv("Forecasted_consumption_18_20.csv", delimiter = ';')
ForecastedConsumption2 = pd.read_csv("Forecasted_consumption_18_20.csv", delimiter = ';')
ForecastedGeneration1 = pd.read_csv("Forecasted_generation_18_20.csv", delimiter = ';')
ForecastedGeneration2 = pd.read_csv("Forecasted_generation_20_22.csv", delimiter = ';')
ActualGeneration1 = pd.read_csv("Actual_generation_18_20.csv", delimiter = ';')
ActualGeneration2 = pd.read_csv("Actual_generation_20_22.csv", delimiter = ';')
DayAhead1 = pd.read_csv("Day-ahead_prices_18_20.csv", delimiter = ';')
DayAhead2 = pd.read_csv("Day-ahead_prices_20_22.csv", delimiter = ';')

#%%
"""
Merging CSVs into 5 dataframes containing each dataset(consumption,generation, etc)
"""

DayAhead = pd.concat([DayAhead1, DayAhead2])
ActualConsumption = pd.concat([ActualConsumption1, ActualConsumption2])
ForecastedConsumption = pd.concat([ForecastedConsumption1, ForecastedConsumption2])
ActualGeneration = pd.concat([ActualGeneration1, ActualGeneration2])
ForecastedGeneration = pd.concat([ForecastedGeneration1, ForecastedGeneration2])


#%%
"""
Because Germany's prices used to be combined with Austria, and Austria split on 1 Oct 2018, the prices need to be merged together
Coerced into floats, then saved over at date.
"""
DayAhead["Germany/Luxembourg[€/MWh]"] = pd.to_numeric(DayAhead["Germany/Luxembourg[€/MWh]"], errors="coerce")
DayAhead["Germany/Austria/Luxembourg[€/MWh]"] = pd.to_numeric(DayAhead["Germany/Austria/Luxembourg[€/MWh]"], errors="coerce")
DayAhead.iloc[:5830,2] = DayAhead.iloc[:5830,15]


#%%
"""
Merge the 5 different dataframes in 1
"""




#%%
plt.plot(DayAhead.iloc[:,2])
plt.show()



#%%
G_DayAhead = DayAheadPrices2[['Date','Time of day','Germany/Luxembourg[€/MWh]']]





















