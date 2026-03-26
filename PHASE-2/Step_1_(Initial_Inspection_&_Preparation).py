print("\n ********************************** PHASE-2 (DATA CLEANING) **********************************\n")

print("--- dataset ---\n")
df_clean=df.copy()
print(df_clean.head())
df_clean.drop('name', axis=1, inplace=True, errors='ignore')
print("--- null values count ---\n")
print(df_clean.isnull().sum())
