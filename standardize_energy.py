# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:48:03 2022

@author: Thomas Pegoraro
"""
#%%
#Standardize all covariates

data = pd.read_csv("Energy_Data.csv", delimiter=",", thousands=',')


data['Wind offshore[MWh]'] = (data['Wind offshore[MWh]']-data['Wind offshore[MWh]'].mean())/data['Wind offshore[MWh]'].std()
data['Wind onshore[MWh]'] = (data['Wind onshore[MWh]']-data['Wind onshore[MWh]'].mean())/data['Wind onshore[MWh]'].std()
data['Photovoltaics[MWh]'] = (data['Photovoltaics[MWh]']-data['Photovoltaics[MWh]'].mean())/data['Photovoltaics[MWh]'].std()
data['Other renewable[MWh]'] = (data['Other renewable[MWh]']-data['Other renewable[MWh]'].mean())/data['Other renewable[MWh]'].std()
data['Nuclear[MWh]'] = (data['Nuclear[MWh]']-data['Nuclear[MWh]'].mean())/data['Nuclear[MWh]'].std()
data['Total (grid load)[MWh]'] = (data['Total (grid load)[MWh]']-data['Total (grid load)[MWh]'].mean())/data['Total (grid load)[MWh]'].std()
