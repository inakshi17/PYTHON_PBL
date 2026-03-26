print("\n----------------------------- Feature Selection -----------------------------\n")

correlation_matrix = df_clean.corr(numeric_only=True)
target_corr = correlation_matrix['HeartDisease'].sort_values(ascending=False)
print("\n--- Features Correlated with Heart Disease ---\n")
print(target_corr)

print("\n--- Variable Correlation Heatmap ---\n")
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BrBG', fmt=".2f", linewidths=0.5)
plt.title("Variable Correlation Heatmap")
plt.show()
important_features = target_corr[abs(target_corr) > 0.01]
print("\n--- Selected Important Features ---\n")
print(important_features)
