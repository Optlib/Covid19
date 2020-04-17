# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings("ignore")

stat ='Daily New Cases'
df = pd.read_csv('Data\italy.csv', usecols= ['Dates',stat])
df['Dates'] = df['Dates'].str.replace(' ','').str.replace('Apr','2020'+'04').str.replace('Mar','2020'+'03').str.replace('Feb','2020'+'02').str.replace('Jan','2020'+'01')
df['Dates']= pd.to_datetime(df['Dates'], format='%Y%m%d') 
df = df.set_index("Dates")


# Learn every 5 days ==> 6th
n_input =5  
n_features = 1

train, test = df[:-n_input], df[-n_input:]
# print(len(train))

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

generator = TimeseriesGenerator(train, train, length=n_input, batch_size=2)
# print(len(generator))


model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit_generator(generator,epochs=50)

pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))


for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    # print(pred_list)
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
index=df[-n_input:].index, columns=['Prediction'])

df_test = pd.concat([df,df_predict], axis=1)
plt.figure(figsize=(20, 5))
plt.plot(df_test.index, df_test[stat])
plt.plot(df_test.index, df_test['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

train = df
scaler.fit(train)
train = scaler.transform(train)
# n_input = 7
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=2)
model.fit_generator(generator,epochs=50)

pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

add_dates = [df.index[-1] + DateOffset(days=x) for x in range(0,n_input+1) ]

future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])

df_proj = pd.concat([df,df_predict], axis=1)
plt.figure(figsize=(20, 5))
plt.plot(df_proj.index, df_proj[stat])
plt.plot(df_proj.index, df_proj['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

 