# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 00:41:00 2020

@author: HB
"""
#from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./Data/uk.csv", usecols =['Dates','Total Deaths'], header=0)
print(df['Total Deaths'])
tunisiaTotalCasesNames=["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28","Mar 29","Mar 30","Mar 31","Apr 01","Apr 02","Apr 03","Apr 04","Apr 05","Apr 06","Apr 07","Apr 08","Apr 09","Apr 10","Apr 11","Apr 12","Apr 13","Apr 14","Apr 15","Apr 16","Apr 17","Apr 18","Apr 19","Apr 20","Apr 21","Apr 22","Apr 23","Apr 24","Apr 25","Apr 26","Apr 27","Apr 28"]

model = pm.auto_arima(df['Total Deaths'], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())
model.plot_diagnostics(figsize=(7,5))
plt.show()


# Forecast
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df['Total Deaths']), len(df['Total Deaths'])+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)

lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df['Dates'], df['Total Deaths'])
plt.plot(fc_series, color='darkgreen')


plt.fill_between(lower_series.index, 
                  lower_series, 
                  upper_series, 
                  color='k', alpha=.15)
plt.xticks(rotation=90)
plt.xticks(tunisiaTotalCasesNames[1::5])

plt.title("UK Covid-19 Forecast")
plt.show()