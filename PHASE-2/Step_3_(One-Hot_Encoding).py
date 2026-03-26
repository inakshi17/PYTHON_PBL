print("\n----------------------------- One-Hot Encoding -----------------------------\n")

print("\n--- Columns before One-Hot Encoding ---\n")
print(df_clean.columns)

categorical_cols = ['thal', 'cp', 'restecg', 'slope']
for col in categorical_cols:
    mode = df_clean[col].mode()[0]
    df_clean[col] = df_clean[col].fillna(mode)
    encoded = pd.get_dummies(df_clean[col], prefix=col, drop_first=True).astype(int)
    df_clean = pd.concat([df_clean, encoded], axis=1)
    df_clean.drop(col, axis=1, inplace=True)
print("\n--- Columns after One-Hot Encoding ---\n")
print(df_clean.columns)
