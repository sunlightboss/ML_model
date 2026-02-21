import  pandas as pd


df = pd.read_csv('/home/nursss/Документы/updated_data.csv')
print(len(df[df["latitude"] == 54.875]))
filtered = df[df["latitude"] == 54.875]
print(filtered["time"].min())
print(filtered["time"].max())