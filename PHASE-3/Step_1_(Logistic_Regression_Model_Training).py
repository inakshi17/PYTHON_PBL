print("\n----------------------------- Logistic Regression Model -----------------------------\n")

lr_model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\n--- Make Predictions ---\n")
print(y_pred_lr)

print("\n--- Logistic Regression Accuracy ---\n")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)
print("\n--- Confusion Matrix ---\n")
print(cm)

print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred_lr))

print("\n--- Confusion Matrix Visualization ---\n")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease','Disease'],
            yticklabels=['No Disease','Disease'])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
