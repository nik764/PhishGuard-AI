
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from url_features import extract_features


print("\n========== LOADING DATASET ==========")

# Load original dataset
df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

print("Dataset loaded successfully!")
print("Total records:", len(df))


print("\n========== EXTRACTING URL FEATURES ==========")

# Extract our own 18 features from every URL
features_list = []

for index, url in enumerate(df["URL"]):

    features = extract_features(str(url))
    features_list.append(features)

    if (index + 1) % 25000 == 0:
        print("Processed:", index + 1, "URLs")


# Convert extracted features into a DataFrame
X = pd.DataFrame(features_list)

# Original dataset labels
y = df["label"]

print("Feature extraction completed!")
print("Feature shape:", X.shape)


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


print("\n========== SAVING MODEL ==========")

# Save the newly trained model separately
joblib.dump(model, "models/phishing_url_model.pkl")

print("New model saved successfully!")
print("Location: models/phishing_url_model.pkl")
