import pandas as pd

df = pd.read_csv('mushroom.csv')

df['poisonous'] = df['poisonous'].map({'p': 1, 'e': 0})

df = pd.get_dummies(df, columns=df.columns.drop('poisonous').values.tolist(), dtype=int)

df.to_csv('mushroom_converted.csv', index=False)
