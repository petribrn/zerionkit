import pandas as pd

df = pd.read_csv('students.csv')

df['Target'] = df['Target'].map({'Graduate': 1, 'Enrolled': 0, 'Dropout': 2})

columns_to_rename = {}
for column in df.columns.values.tolist():
    columns_to_rename[column] = column.lower().strip()

df.rename(columns=columns_to_rename, inplace=True)
df.to_csv('students_converted.csv', index=False)
