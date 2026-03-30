print("\n--- KNN Accuracy Analysis for Different K Values ---\n")
accuracy_list = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)
    print(f"K={k}, Accuracy={round(acc, 2)}")
best_k = accuracy_list.index(max(accuracy_list)) + 1
print("\nBest K:", best_k)
print("Best Accuracy:", round(max(accuracy_list), 2))

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
print("\n--- knn Make Predictions ---\n")
y_pred_knn = knn_model.predict(X_test)

print("\n--- KNN Accuracy ---\n")
print("Final KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

print("\n--- Confusion Matrix ---\n")
print(confusion_matrix(y_test, y_pred_knn))

print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)

print("\n--- Confusion Matrix Visualization ---\n")
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Disease','Disease'],
            yticklabels=['No Disease','Disease'])
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
