print("\n----------------------------- Outlier Deletion -----------------------------\n")

print(f"Dataset Shape before Deletion of outliers: {df_clean.shape}")
cols_to_clean = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in cols_to_clean:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[(df_clean[col] >= Q1 - 1.5*IQR) & (df_clean[col] <= Q3 + 1.5*IQR)]
df_clean.reset_index(drop=True, inplace=True)
print(f"Dataset Shape after Deletion of outliers: {df_clean.shape}")
