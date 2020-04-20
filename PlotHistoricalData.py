import  matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./Data/Covid19_Data.csv", sep=';')
df= df[df['Country']=='UK']

plt.subplots(figsize=(10,4))
plt.plot( df['Dates'], df['TotalCases'], color='r')
plt.xticks(rotation=45)
plt.xticks(df['Dates'][1::5])


plt.grid(axis='x')
plt.legend(title='UK Total Cases :')
plt.title('Covid-19 Total cases')

plt.show()
