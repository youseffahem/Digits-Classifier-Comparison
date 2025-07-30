from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

svc_acc = accuracy_score(y_test, svc_pred)
tree_acc = accuracy_score(y_test, tree_pred)

svc_report = classification_report(y_test, svc_pred)
tree_report = classification_report(y_test, tree_pred)
print (f"svc_report = {svc_report}")
print (f"tree_report = {svc_report}")

print("SVC Model==")
print(f"Accuracy: {svc_acc:.4f}")
print("Decision Tree Model==")
print(f"Accuracy: {tree_acc:.4f}")

with open("result.txt", "w") as f:
    f.write("=== SVC Model ===\n")
    f.write(f"Accuracy: {svc_acc:.4f}\n")
    f.write(svc_report + "\n")

    f.write("=== Decision Tree Model ===\n")
    f.write(f"Accuracy: {tree_acc:.4f}\n")
    f.write(tree_report)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, svc_pred), annot=True, fmt='d', cmap='Blues')
plt.title("SVC Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, tree_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()