
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

probabilities = model.predict_proba(X_test)

# Random Forest classes are [0=Phishing, 1=Legitimate]
results = pd.DataFrame({
    "URL": urls_test,
    "Actual": y_test,
    "Phishing_Confidence": probabilities[:, 0],
    "Legitimate_Confidence": probabilities[:, 1]
}, index=y_test.index)

results = results.merge(
    urls[["URL", "source"]],
    on="URL",
    how="left"
)

# Exclusively Phish360 legitimate false positives
errors = results[
    (results["source"] == "Phish360") &
    (results["Actual"] == 1) &
    (results["Phishing_Confidence"] > results["Legitimate_Confidence"])
].copy()

print("\n========== PREDICTION CONFIDENCE ==========")

print("Total false positives:", len(errors))

print("\nAverage phishing confidence:",
      round(errors["Phishing_Confidence"].mean() * 100, 2), "%")

print("Average legitimate confidence:",
      round(errors["Legitimate_Confidence"].mean() * 100, 2), "%")

print("\n========== CONFIDENCE GROUPS ==========")

for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:

    count = (
        errors["Phishing_Confidence"] >= threshold
    ).sum()

    print(
        f"Phishing confidence >= {threshold * 100:.0f}%:",
        count
    )

print("\n========== SAMPLE RESULTS ==========")

for _, row in errors.head(15).iterrows():

    print("\nURL:", row["URL"])
    print("Phishing confidence:",
          round(row["Phishing_Confidence"] * 100, 2), "%")
    print("Legitimate confidence:",
          round(row["Legitimate_Confidence"] * 100, 2), "%")

print("\nAnalysis completed!")
