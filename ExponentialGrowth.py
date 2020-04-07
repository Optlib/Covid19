# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 00:41:00 2020

@author: HB
"""

import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

dataset = pd.read_csv("./Data/uk.csv", usecols =['Dates','Total Cases'], header=0)
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)


Y=dataset['Total Cases']
X=np.arange(0.,  len(Y), 1.0)


# Fit and summarize OLS model
mod = sm.OLS(np.log(Y), X)
res = mod.fit()
print(res.summary())

plt.plot(X,  0.2344 *X, 'r--', X, np.log(Y))
plt.show()
