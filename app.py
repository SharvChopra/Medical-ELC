import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils import load_images, extract_hog_features

# Load dataset
tumor_imgs, tumor_labels = load_images("brain_tumor_dataset/yes", 1)
normal_imgs, normal_labels = load_images("brain_tumor_dataset/no", 0)

X = np.array(tumor_imgs + normal_imgs)
y = np.array(tumor_labels + normal_labels)

print("Total images:", len(X))

# Feature extraction
X_features = extract_hog_features(X)

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_features, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Train model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

# Training vs Validation Accuracy
print("Training Accuracy:", model.score(X_train, y_train))
print("Validation Accuracy:", model.score(X_val, y_val))

# Test evaluation
y_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")
