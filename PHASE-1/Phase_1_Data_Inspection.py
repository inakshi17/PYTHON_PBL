import pandas as pd
df = pd.read_csv("heart.csv")

print("--- Dataset Shape (Rows, Columns) ---")
print(df.shape)
print("\n")

print("--- Dataset Information ---")
print(df.info())
print("\n")

print("--- Dataset Preview (Head) ---")
print(df.head())
print("\n")

print("--- Statistical Summary ---")
print(df.describe())
