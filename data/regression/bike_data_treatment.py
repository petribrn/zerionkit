import pandas as pd

df = pd.read_csv('bike.csv')

df = df.drop(columns=['instant', 'dteday', 'yr'])

df = pd.get_dummies(df, columns=['season', 'mnth', 'hr' ,'weekday', 'weathersit']).astype(int)

df = df[[col for col in df.columns if col not in ['casual', 'registered', 'cnt']]+ ['casual', 'registered', 'cnt']]

df.to_csv('bike_converted.csv', index=False)
