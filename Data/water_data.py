import pandas as pd

df = pd.read_csv('/home/nursss/Загрузки/Smart_irrigation_dataset.csv')
print(df.head(10))
df_half = df.sample(frac=0.5, random_state=42)  # случайная половина
df_half = df_half.to_csv('half_water.csv', index=True)