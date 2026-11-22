
import pandas as pd
import joblib

from urllib.parse import urlsplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n========== LOADING DATA ==========")

urls = pd.read_csv("dataset/combined_urls.csv")
features = pd.read_csv("dataset/combined_features.csv")

X = features.drop(columns=["label"])
y = features["label"]

print("Total URLs:", len(urls))
print("Total features:", X.shape[1])


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
    return "Homepage"


print("\n========== SPLITTING DATA ==========")

X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
    X,
    y,
    urls["URL"],
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_sources = urls.loc[X_train.index, "source"]

train_data = pd.DataFrame({
    "source": train_sources,
    "category": urls_train.apply(get_category),
    "label": y_train
}, index=X_train.index)


print("\n========== CALCULATING SOURCE-AWARE WEIGHTS ==========")

group_sizes = train_data.groupby(
    ["source", "category", "label"]
)["label"].transform("size")

maximum_group_size = group_sizes.max()

weights = (maximum_group_size / group_sizes) ** 0.5

# Prevent very small groups from receiving excessive influence
weights = weights.clip(upper=3.0)

# Normalize weights for easier comparison
weights = weights / weights.mean()

print("Number of training groups:",
      train_data.groupby(["source", "category", "label"]).ngroups)

print("Average training weight:", round(weights.mean(), 3))
print("Minimum training weight:", round(weights.min(), 3))
print("Maximum training weight:", round(weights.max(), 3))


print("\n========== TRAINING SOURCE-AWARE MODEL ==========")

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

print("Training completed!")


print("\n========== OVERALL EVALUATION ==========")

predictions = model.predict(X_test)

print("Accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"]
))


print("\n========== PHISH360 LEGITIMATE EVALUATION ==========")

test_data = pd.DataFrame({
    "URL": urls_test,
    "Actual": y_test,
    "Predicted": predictions
}, index=y_test.index)

test_data = test_data.merge(
    urls[["URL", "source"]],
    on="URL",
    how="left"
)

phish360_legitimate = test_data[
    (test_data["source"] == "Phish360") &
    (test_data["Actual"] == 1)
]

correct = (
    phish360_legitimate["Predicted"] == 1
).sum()

total = len(phish360_legitimate)

print("Total Phish360 legitimate URLs:", total)
print("Correctly accepted:", correct)
print("Incorrectly flagged:", total - correct)

if total:
    print("Legitimate recall:",
          round(correct / total * 100, 2), "%")


print("\n========== SAVING MODEL ==========")

joblib.dump(
    model,
    "models/phishing_url_model_source_balanced.pkl"
)

print("Model saved successfully!")
print("models/phishing_url_model_source_balanced.pkl")
