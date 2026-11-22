
import pandas as pd
import joblib

from urllib.parse import urlsplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n========== LOADING DATA ==========")

df = pd.read_csv("dataset/combined_urls.csv")
features = pd.read_csv("dataset/combined_features.csv")

X = features.drop(columns=["label"])
y = features["label"]

print("Total URLs:", len(df))
print("Total features:", X.shape[1])


# Identify URL categories
def get_category(url):
    parsed = urlsplit(str(url))

    has_path = bool(parsed.path.strip("/"))
    has_query = bool(parsed.query)

    if has_path and has_query:
        return "Path + Query"
    elif has_path:
        return "Path"
    elif has_query:
        return "Query"
    else:
        return "Homepage"


df["category"] = df["URL"].apply(get_category)


print("\n========== SPLITTING DATA ==========")

X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
    X,
    y,
    df["URL"],
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== CALCULATING TRAINING WEIGHTS ==========")

train_categories = urls_train.apply(get_category)

weights = pd.Series(1.0, index=X_train.index)

for category in train_categories.unique():

    category_mask = train_categories == category
    category_labels = y_train[category_mask]

    counts = category_labels.value_counts()

    if len(counts) == 2:

        majority_count = counts.max()

        for label, count in counts.items():

            weight = min(
                (majority_count / count) ** 0.5,
                5.0
            )

            # Corrected index selection
            matching_indices = category_labels.index[
                category_labels == label
            ]

            weights.loc[matching_indices] = weight

    print(
        category,
        "label counts:",
        counts.to_dict()
    )

print("\nTraining weights calculated.")


print("\n========== TRAINING WEIGHTED RANDOM FOREST ==========")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    sample_weight=weights
)

print("Model training completed!")


print("\n========== MODEL EVALUATION ==========")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"]
    )
)


print("\n========== SAVING WEIGHTED MODEL ==========")

joblib.dump(
    model,
    "models/phishing_url_model_weighted.pkl"
)

print("Model saved successfully!")
print("Location: models/phishing_url_model_weighted.pkl")
