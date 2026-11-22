
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n========== LOADING COMBINED FEATURES ==========")

df = pd.read_csv("dataset/combined_features.csv")

X = df.drop(columns=["label"])
y = df["label"]

print("Total records:", len(df))
print("Total features:", X.shape[1])

print("\nLabel distribution:")
print(y.value_counts().sort_index())


print("\n========== SPLITTING DATA ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


print("\n========== TRAINING RANDOM FOREST ==========")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")


print("\n========== MODEL EVALUATION ==========")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"]
    )
)


print("\n========== SAVING NEW MODEL ==========")

joblib.dump(
    model,
    "models/phishing_url_model_combined.pkl"
)

print("New model saved successfully!")
print("Location: models/phishing_url_model_combined.pkl")
