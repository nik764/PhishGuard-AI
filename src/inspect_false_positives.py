
import pandas as pd
import joblib

from urllib.parse import urlsplit
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

false_positive_mask = (
    (y_test == 1) & (predictions == 0)
)

false_positives = pd.DataFrame({
    "URL": urls_test[false_positive_mask],
    "Category": urls_test[false_positive_mask].apply(
        lambda url: (
            "Path + Query"
            if urlsplit(str(url)).path.strip("/")
            and urlsplit(str(url)).query
            else "Path"
            if urlsplit(str(url)).path.strip("/")
            else "Query"
            if urlsplit(str(url)).query
            else "Homepage"
        )
    )
})

# Attach original dataset source information
source_info = urls[["URL", "source"]].drop_duplicates("URL")

false_positives = false_positives.merge(
    source_info,
    on="URL",
    how="left"
)

print("\n========== FALSE POSITIVE SOURCE DISTRIBUTION ==========")
print(false_positives["source"].value_counts(dropna=False))

print("\n========== EXAMPLES BY CATEGORY AND SOURCE ==========")

for category in ["Homepage", "Path", "Query", "Path + Query"]:

    category_data = false_positives[
        false_positives["Category"] == category
    ]

    print(f"\n--- {category.upper()} ---")

    for _, row in category_data.head(10).iterrows():
        print("URL:", row["URL"])
        print("Source:", row["source"])
        print()

false_positives.to_csv(
    "dataset/weighted_false_positives_with_sources.csv",
    index=False
)

print("\nSaved: dataset/weighted_false_positives_with_sources.csv")
