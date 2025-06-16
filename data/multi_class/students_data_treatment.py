import pandas as pd

df = pd.read_csv('students.csv')

df['Target'] = df['Target'].map({'Graduate': 1, 'Enrolled': 0, 'Dropout': 2})

df.to_csv('students_converted.csv', index=False)
