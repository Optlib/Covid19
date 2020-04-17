# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 00:41:00 2020

@author: HB
"""

import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

dataset = pd.read_csv("./Data/france.csv", usecols =['Dates','Total Cases'], header=0)
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)


Y=dataset['Total Cases']
X=np.arange(0.,  len(Y), 1.0)


# Fit and summarize OLS model


logY=np.log(np.where(Y==0, 0.000000001, Y))
print(logY)
mod = sm.OLS(logY, X)
res = mod.fit()
print(res.summary())

plt.plot(X,  0.2645 *X, 'r--', X, logY)
plt.show()
