import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./Data/Covid19_Data.csv", sep=';')
groups=df.groupby(['Country'])
print(groups.groups.keys())
df= df[df['Country'].isin({'UK', 'France', 'Italy', 'Spain', 'US'})]

df.groupby('Country')['TotalDeaths'].agg('sum').plot(kind='pie',autopct='%1.1f%%',
                                                      shadow=True, startangle=0,
                                                      explode=(0, 0,0.0, 0.0,0.2),
                                                      title='Group-By Country')

fig, ax = plt.subplots(figsize=(10,4))
groups=df.groupby(['Country'])
for key, grp in groups:
    ax.plot(grp['Dates'], grp['TotalDeaths'], label =df['Dates'][1::5])
    # plt.xticks(df['Dates'][1::5])
    plt.xticks(rotation=45)




plt.show()
