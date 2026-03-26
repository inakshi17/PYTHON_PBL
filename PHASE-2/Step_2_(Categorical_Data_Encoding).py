print("\n ----------------------------- Categorical Mapping & Cleaning -----------------------------\n")

df_clean['sex'] = df_clean['sex'].replace({ 'Male':1,'Female':0,'M':1,'F':0,'female':0,'male':1,'f':0,'m':1 })
df_clean['sex'] = pd.to_numeric(df_clean['sex'], errors='coerce')
df_clean['sex'] = df_clean['sex'].fillna(df_clean['sex'].mode()[0])
print("\n--- clean sex column ---\n")
print(df_clean['sex'].head())

df_clean['exang'] = df_clean['exang'].replace({'yes':1,'no':0,'Yes':1,'No':0 })
df_clean['exang'] = pd.to_numeric(df_clean['exang'], errors='coerce')
df_clean['exang'] = df_clean['exang'].fillna(df_clean['exang'].mode()[0])
print("\n--- clean exercise-induced angina column ---\n")
print(df_clean['exang'].head())

df_clean['fbs'] = pd.to_numeric(df_clean['fbs'], errors='coerce')
df_clean['fbs'] = (df_clean['fbs'] >= 120).astype(int)
df_clean['fbs'] = df_clean['fbs'].fillna(df_clean['fbs'].mode()[0])
print("\n--- clean asting blood sugar column ---\n")
print(df_clean['fbs'].head())
