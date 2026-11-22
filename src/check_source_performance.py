
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

print("\n========== LOADING DATA ==========")

urls = pd.read_csv("dataset/combined_urls.csv")
features = pd.read_csv("dataset/combined_features.csv")

X = features.drop(columns=["label"])
y = features["label"]

X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
    X,
    y,
    urls["URL"],
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = joblib.load("models/phishing_url_model_weighted.pkl")

predictions = pd.Series(
    model.predict(X_test),
    index=y_test.index
)

# Match source information to the test URLs
test_data = pd.DataFrame({
    "URL": urls_test,
    "Actual": y_test,
    "Predicted": predictions
})

test_data = test_data.merge(
    urls[["URL", "source"]],
    on="URL",
    how="left"
)

print("\n========== SOURCE-WISE LEGITIMATE PERFORMANCE ==========")

legitimate = test_data[test_data["Actual"] == 1]

for source, group in legitimate.groupby("source"):

    total = len(group)
    accepted = (group["Predicted"] == 1).sum()
    flagged = (group["Predicted"] == 0).sum()

    recall = accepted / total * 100 if total else 0

    print(f"\nSource: {source}")
    print("Total legitimate URLs:", total)
    print("Correctly accepted:", accepted)
    print("Incorrectly flagged:", flagged)
    print("Legitimate recall:", round(recall, 2), "%")

print("\n========== CHECK COMPLETED ==========")
