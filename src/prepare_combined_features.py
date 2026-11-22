
import pandas as pd
from url_features import extract_features

print("\n========== LOADING COMBINED DATASET ==========")

df = pd.read_csv("dataset/combined_urls.csv")

print("Total URLs:", len(df))

print("\n========== EXTRACTING 19 FEATURES ==========")

features_list = []

for index, url in enumerate(df["URL"]):

    features = extract_features(str(url))
    features_list.append(features)

    if (index + 1) % 10000 == 0:
        print("Processed:", index + 1, "URLs")

X = pd.DataFrame(features_list)
y = df["label"]

print("\n========== FEATURE EXTRACTION RESULTS ==========")

print("Feature shape:", X.shape)
print("Number of labels:", len(y))
print("Feature columns:")
print(X.columns.tolist())

# Save features and labels separately
X["label"] = y.values

X.to_csv("dataset/combined_features.csv", index=False)

print("\nCombined feature dataset saved!")
print("Location: dataset/combined_features.csv")
