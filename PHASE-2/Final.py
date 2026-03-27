import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("\n\n\n\n ********************************** PHASE-2 (DATA CLEANING) **********************************\n\n\n\n")

df = pd.read_csv("heart.csv")
print("--- dataset ---\n")
df_clean=df.copy()
print(df_clean.head())
df_clean.drop('name', axis=1, inplace=True, errors='ignore')
print("--- null values count ---\n")
print(df_clean.isnull().sum())


print("\n ----------------------------- Categorical Mapping & Cleaning -----------------------------\n")

df_clean['sex'] = df_clean['sex'].replace({
    'Male':1,'Female':0,'M':1,'F':0,
    'female':0,'male':1,'f':0,'m':1
})
df_clean['sex'] = pd.to_numeric(df_clean['sex'], errors='coerce')
df_clean['sex'] = df_clean['sex'].fillna(df_clean['sex'].mode()[0])
print("\n--- clean sex column ---\n")
print(df_clean['sex'].head())

df_clean['exang'] = df_clean['exang'].replace({
    'yes':1,'no':0,'Yes':1,'No':0
})
df_clean['exang'] = pd.to_numeric(df_clean['exang'], errors='coerce')
df_clean['exang'] = df_clean['exang'].fillna(df_clean['exang'].mode()[0])
print("\n--- clean exercise-induced angina column ---\n")
print(df_clean['exang'].head())

df_clean['fbs'] = pd.to_numeric(df_clean['fbs'], errors='coerce')
df_clean['fbs'] = (df_clean['fbs'] >= 120).astype(int)
df_clean['fbs'] = df_clean['fbs'].fillna(df_clean['fbs'].mode()[0])
print("\n--- clean fasting blood sugar column ---\n")
print(df_clean['fbs'].head())


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
