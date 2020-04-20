import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("./Data/Covid19_Data.csv", sep=';')
df= df[df['Country']=='UK']
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)


Y=df['TotalCases']
X=np.arange(0.,  len(Y), 1.0)


# Fit and summarize OLS model


logY=np.log(np.where(Y==0, 0.000000001, Y))
print(logY)
mod = sm.OLS(logY, X)
res = mod.fit()
print(res.summary())

plt.plot(X,  0.2645 *X, 'r--', X, logY)
plt.show()
