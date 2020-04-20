import pmdarima as pm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./Data/Covid19_Data.csv", sep=';')
df= df[df['Country']=='UK']

model = pm.auto_arima(df['TotalDeaths'], start_p=1, start_q=1,
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
index_of_fc = np.arange(len(df['TotalDeaths']), len(df['TotalDeaths'])+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)

lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df['Dates'], df['TotalDeaths'])
plt.plot(fc_series, color='darkgreen')


plt.fill_between(lower_series.index, 
                  lower_series, 
                  upper_series, 
                  color='k', alpha=.15)
plt.xticks(rotation=45)
plt.xticks(df['Dates'][1::5])

plt.title("UK Covid-19 Forecast")
plt.show()