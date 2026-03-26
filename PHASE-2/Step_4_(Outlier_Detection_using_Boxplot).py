print("\n----------------------------- Outlier Detection -----------------------------\n")

cols_to_force = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'HeartDisease']
for col in cols_to_force:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
cols_to_check = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
plt.figure(figsize=(10,6))
sns.boxplot(data=df_clean[cols_to_check])
plt.title('Outlier Detection')
plt.ylabel('Values')
plt.xlabel('features')
plt.show()
