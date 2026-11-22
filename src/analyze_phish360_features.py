
import pandas as pd

print("\n========== LOADING PHISH360 DATA ==========")

df = pd.read_csv("dataset/combined_features.csv")

urls = pd.read_csv("dataset/combined_urls.csv")

data = pd.concat(
    [
        urls[["URL", "source", "label"]].reset_index(drop=True),
        df.drop(columns=["label"]).reset_index(drop=True)
    ],
    axis=1
)

# Keep only records exclusively from Phish360
phish360 = data[data["source"] == "Phish360"].copy()

print("Total Phish360 records:", len(phish360))

print("\n========== LABEL DISTRIBUTION ==========")
print(phish360["label"].value_counts())

feature_columns = [
    column for column in df.columns
    if column != "label"
]

print("\n========== FEATURE AVERAGES ==========")

comparison = phish360.groupby("label")[feature_columns].mean().T

comparison.columns = ["Phishing (0)", "Legitimate (1)"]

comparison["Difference"] = (
    comparison["Legitimate (1)"] -
    comparison["Phishing (0)"]
)

print(comparison.round(3).to_string())

print("\n========== URL LENGTH COMPARISON ==========")

print(
    phish360.groupby("label")["URL"]
    .apply(lambda x: x.str.len().mean())
)

print("\nAnalysis completed!")
