print("\n----------------------------- Feature Standardization -----------------------------\n")

selected_cols = important_features.index.drop('HeartDisease')
X = df_clean[selected_cols]
y = df_clean['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
cols_to_scale = ['oldpeak','thalach','trestbps','age','chol']
cols_to_scale = [col for col in cols_to_scale if col in X_train.columns]
if cols_to_scale:
    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])
print("\n--- columns after Standardization ---\n")
print(X_train[cols_to_scale].head())
