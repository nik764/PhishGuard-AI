
import pandas as pd
import joblib

from urllib.parse import urlsplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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


categories = urls_test.apply(get_category)

models = {
    "Weighted Model": (
        "models/phishing_url_model_weighted.pkl",
        X_test
    ),
    "Weighted No-HTTPS Model": (
        "models/phishing_url_model_weighted_no_https.pkl",
        X_test.drop(columns=["IsHTTPS"])
    ),
    "Weighted No-HTTPS No-DomainLength Model": (
        "models/phishing_url_model_weighted_no_https_domain.pkl",
        X_test.drop(columns=["IsHTTPS", "DomainLength"])
    )
}

print("\n========== FINAL THREE-MODEL COMPARISON ==========")

for model_name, (model_path, model_features) in models.items():

    model = joblib.load(model_path)
    predictions = model.predict(model_features)

    print(f"\n\n========== {model_name.upper()} ==========")

    print(
        "Overall accuracy:",
        round(accuracy_score(y_test, predictions) * 100, 2),
        "%"
    )

    for category in ["Homepage", "Path", "Query", "Path + Query"]:

        mask = categories == category

        actual = y_test[mask]
        predicted = predictions[mask]

        matrix = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        )

        tn, fp, fn, tp = matrix.ravel()

        legitimate_recall = (
            tp / (tp + fn) * 100 if (tp + fn) else 0
        )

        phishing_recall = (
            tn / (tn + fp) * 100 if (tn + fp) else 0
        )

        print(f"\n{category}:")
        print("Total:", len(actual))
        print("Accuracy:", round(
            accuracy_score(actual, predicted) * 100, 2
        ), "%")
        print("Legitimate URLs correctly accepted:", tp)
        print("Legitimate URLs incorrectly flagged:", fn)
        print("Phishing URLs correctly detected:", tn)
        print("Phishing URLs missed:", fp)
        print("Legitimate recall:", round(legitimate_recall, 2), "%")
        print("Phishing recall:", round(phishing_recall, 2), "%")

print("\n========== FINAL COMPARISON COMPLETED ==========")
