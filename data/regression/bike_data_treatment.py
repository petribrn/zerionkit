import pandas as pd

df = pd.read_csv('bike.csv')

df = df.drop(columns=['instant', 'dteday'])

df.to_csv('bike_converted.csv', index=False)
