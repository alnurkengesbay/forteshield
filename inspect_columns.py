import pandas as pd
import os

data_dir = r"C:\Users\alnur\Documents\foreshield ai\data"
patterns_file = os.path.join(data_dir, "поведенческие_паттерны_клиентов_3.csv")

try:
    df = pd.read_csv(patterns_file, sep=';', encoding='cp1251')
    print("Columns in patterns file:")
    print(df.columns.tolist())
except Exception as e:
    print(e)
