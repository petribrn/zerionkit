import pandas as pd

df = pd.read_csv('students.csv')

columns_to_rename = {}
for column in df.columns.values.tolist():
    columns_to_rename[column] = column.lower().strip()

df.rename(columns=columns_to_rename, inplace=True)

df = pd.get_dummies(df, columns=['marital status', 'application mode', 'application order','course', 'daytime/evening attendance',
                                 'daytime/evening attendance', 'previous qualification', "mother's qualification", "father's qualification",
                                 "mother's occupation", "father's occupation", "nacionality", "target"],dtype=int)

df.to_csv('students_converted.csv', index=False)
