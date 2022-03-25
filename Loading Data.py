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

import datetime as dtm

import random 
from dateutil.parser import parse
from datetime import date
from datetime import datetime, timedelta
from workalendar.europe import Germany
from pandas.plotting import table

print('done')


#%%
"""
reading all relevant CSV
"""
ActualConsumption1 = pd.read_csv("Actual_consumption_18_20.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
ActualConsumption2 = pd.read_csv("Actual_consumption_20_22.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
print("Actual Consumption")
ForecastedConsumption1 = pd.read_csv("Forecasted_consumption_18_20.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
ForecastedConsumption2 = pd.read_csv("Forecasted_consumption_20_22.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
print("Forecasted Consumption")
ForecastedGeneration1 = pd.read_csv("Forecasted_generation_18_20.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
ForecastedGeneration2 = pd.read_csv("Forecasted_generation_20_22.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
print("Forecasted Generation")
ActualGeneration1 = pd.read_csv("Actual_generation_18_20.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
ActualGeneration2 = pd.read_csv("Actual_generation_20_22.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
print("Actual Generation")
DayAhead1 = pd.read_csv("Day-ahead_prices_18_20.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
DayAhead2 = pd.read_csv("Day-ahead_prices_20_22.csv", delimiter = ';', parse_dates = [["Date", "Time of day"]])
print("Day Ahead")


#%%
"""
Merging CSVs into 5 seperate dataframes containing each dataset(consumption,generation, etc)
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
DayAhead.iloc[:5830,1] = DayAhead.iloc[:5830,14]


#%%



plt.plot(DayAhead.iloc[:,1])
plt.show()



#%%
""" Adding Day & Holidays """

G_DayAhead = DayAhead.iloc[:,0:2]

#%%
G_DayAhead['Day of Week'] = G_DayAhead['Date_Time of day'].dt.day_of_week
G_DayAhead.set_index('Date_Time of day', inplace = True)


#%%

def weekday_definition(orig_weekday_type):
    if orig_weekday_type == 0:
        return 'Monday'
    elif orig_weekday_type >=1 and orig_weekday_type <=3:
        return 'Tuesday-Thursday'
    elif orig_weekday_type == 4:
        return 'Friday'
    elif orig_weekday_type == 5:
        return 'Saturday'
    elif orig_weekday_type == 6:
        return 'Sunday'
    elif orig_weekday_type == 7:
        return 'Holiday'
#%%

calDe = Germany()
HolidaysDe = []

sYear = G_DayAhead.index[0].year
eYear = G_DayAhead.index[-1].year
for y in np.arange(sYear,eYear+1):
    tempDe = calDe.holidays(y)
    
    HolidaysDe = HolidaysDe + tempDe

#%%
HolidaysDe # 2-dim list

#%%
Holidays_datesDe = [sl[0] for sl in HolidaysDe] #1-dim list

#%%

#Marking holidays in spot price data frame
for T in range(0,len(G_DayAhead.index)):
    if G_DayAhead.index.date[T] in Holidays_datesDe: 
        G_DayAhead.iloc[T,1] = 7 
    else:
        G_DayAhead.iloc[T,1] = G_DayAhead.iloc[T,1]


print('Done')


#%%
'''
holiday_TF = []
for T in range(0,len(G_DayAhead.index)+1):
    if G_DayAhead.index.date[T] in Holidays_datesDe: 
        holiday_TF[T] = 1
    else:
        holiday_TF[T] = 0
print('Done')
'''

#%%%
"""
Merge the 5 different dataframes into 2
"""

dataframe_list = [ActualConsumption, ActualGeneration, ForecastedConsumption, ForecastedGeneration]

G_DayAhead = G_DayAhead[~G_DayAhead.index.duplicated(keep ="first")]
#df3 = df3[~df3.index.duplicated(keep='first')]

#%%
Merged_Data_Forecasted = pd.merge(
    G_DayAhead,
    ForecastedConsumption,
    how="inner",
    on=['Date_Time of day']
)

Merged_Data_Forecasted =pd.merge(
    Merged_Data_Forecasted,
    ForecastedGeneration,
    how="inner",
    on=['Date_Time of day']
)
Dat
Merged_Data_Actual =pd.merge(
    G_DayAhead,
    ActualConsumption,
    how="inner",
    on=['Date_Time of day']
)

Merged_Data_Actual =pd.merge(
    Merged_Data_Actual,
    ActualGeneration,
    how="inner",
    on=['Date_Time of day']
)


#%%
dayahead_duplicates = G_DayAhead[G_DayAhead.index.duplicated(keep = False)]
not_duplicated_rows_Actual = ~Merged_Data_Actual.duplicated(subset='Date_Time of day',keep='first')
not_duplicated_rows_Forecasted = ~Merged_Data_Forecasted.duplicated(subset='Date_Time of day',keep='first')

#%%
#investigating 4 duplicates in day ahead
#10 28 2018

view_ddups = G_DayAhead.loc[dayahead_duplicates.index[1]:,]
view_ddups = G_DayAhead.loc['2018-10-28 00:00:00':,]

#%%

#checking to see if there are missing hours in day ahead
missing_hours = pd.date_range(start = '2018-01-31', end = '2022-01-31', freq="H" ).difference(G_DayAhead.index)




#%%
#view duplicated rows in merge

mda_duplicate_rows = Merged_Data_Actual[Merged_Data_Actual.duplicated(subset='Date_Time of day',keep='first')]
#%%
Merged_Data_Actual = Merged_Data_Actual[not_duplicated_rows_Actual.values]
Merged_Data_Forecasted = Merged_Data_Forecasted[not_duplicated_rows_Forecasted]


#%%
#DataFrame.duplicated(subset=None, keep='first')

""" Export Data set """
Merged_Data_Actual.to_csv('Data Actual.csv', index=False)
Merged_Data_Forecasted.to_csv('Data Forecasted.csv', index=False)

#%%
#spot checks
random_samples =random.sample(range(0, len(Merged_Data_Actual)), 10)

sample_actual_merged = Merged_Data_Actual.iloc[random_samples,]
sample_actualconsumption = ActualConsumption[ActualConsumption['Date_Time of day'].isin(sample_actual_merged['Date_Time of day'])]
sample_actualgeneration = ActualGeneration[ActualGeneration['Date_Time of day'].isin(sample_actual_merged['Date_Time of day'])]

sample_forecasted_merged = Merged_Data_Forecasted.iloc[random_samples,]
sample_forecastedconsumption = ForecastedConsumption[ForecastedConsumption['Date_Time of day'].isin(sample_forecasted_merged['Date_Time of day'])]
sample_forecastedgeneration = ForecastedGeneration[ForecastedGeneration['Date_Time of day'].isin(sample_forecasted_merged['Date_Time of day'])]
    







