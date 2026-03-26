print("\n----------------------------- missing value imputation -----------------------------\n")

cols_to_median = ['oldpeak', 'trestbps']
df_clean[cols_to_median] = df_clean[cols_to_median].fillna(df_clean[cols_to_median].median())
df_clean['ca'] = df_clean['ca'].fillna(df_clean['ca'].mode()[0])
df_clean['HeartDisease'] = df_clean['HeartDisease'].fillna(df_clean['HeartDisease'].mode()[0])
thalach_mean = df_clean['thalach'].mean()
df_clean['thalach'] = df_clean['thalach'].fillna(thalach_mean)
print("\n--- missing value imputation ---\n")
print(df_clean[['oldpeak', 'trestbps', 'sex', 'ca', 'thalach']].head())

df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
print("\n--- null values count ---\n")
print(df_clean.isnull().sum())
