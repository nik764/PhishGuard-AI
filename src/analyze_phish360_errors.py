
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

test_data = X_test.copy()

test_data["URL"] = urls_test
test_data["Actual"] = y_test
test_data["Predicted"] = predictions

test_data = test_data.merge(
    urls[["URL", "source"]],
    on="URL",
    how="left"
)

# Only exclusively Phish360 records
phish360 = test_data[
    test_data["source"] == "Phish360"
].copy()

# Legitimate URLs incorrectly predicted as phishing
errors = phish360[
    (phish360["Actual"] == 1) &
    (phish360["Predicted"] == 0)
].copy()

correct = phish360[
    (phish360["Actual"] == 1) &
    (phish360["Predicted"] == 1)
].copy()

print("\n========== PHISH360 LEGITIMATE COMPARISON ==========")

print("Incorrectly flagged:", len(errors))
print("Correctly accepted:", len(correct))

columns = [
    "URLLength",
    "DomainLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfDegitsInURL",
    "IsHTTPS",
    "DomainNameEntropy"
]

print("\nAverage features of incorrectly flagged URLs:")
print(errors[columns].mean().round(3))

print("\nAverage features of correctly accepted URLs:")
print(correct[columns].mean().round(3))

print("\n========== ERROR URL EXAMPLES ==========")

for url in errors["URL"].head(20):
    print(url)

print("\nAnalysis completed!")
