
import pandas as pd

print("\n========== PREPARING COMBINED DATASET ==========")

# 1. Load PhiUSIIL
print("\nLoading PhiUSIIL...")

phi = pd.read_csv(
    "dataset/PhiUSIIL_Phishing_URL_Dataset.csv",
    usecols=["URL", "label"]
)

phi = phi.rename(columns={"label": "label"})
phi["source"] = "PhiUSIIL"


# 2. Load LegitPhish
print("Loading LegitPhish...")

legit = pd.read_csv(
    "dataset/url_features_extracted1.csv",
    usecols=["URL", "ClassLabel"]
)

legit["label"] = pd.to_numeric(
    legit["ClassLabel"], errors="coerce"
)

legit = legit[["URL", "label"]]
legit["source"] = "LegitPhish"


# 3. Load Phish360
print("Loading Phish360...")

phish360_legit = pd.read_parquet(
    "dataset/Phish360/Phish360_legit.parquet",
    columns=["URL"]
)

phish360_legit["label"] = 1
phish360_legit["source"] = "Phish360"

phish360_phish = pd.read_parquet(
    "dataset/Phish360/Phish360_phish.parquet",
    columns=["URL"]
)

phish360_phish["label"] = 0
phish360_phish["source"] = "Phish360"


# 4. Combine all datasets
print("\nCombining datasets...")

df = pd.concat(
    [
        phi[["URL", "label", "source"]],
        legit[["URL", "label", "source"]],
        phish360_legit,
        phish360_phish
    ],
    ignore_index=True
)

# Clean URLs and labels
df["URL"] = df["URL"].astype("string").str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df.dropna(subset=["URL", "label"])
df = df[df["URL"] != ""]

df["label"] = df["label"].astype(int)

# Keep only valid binary labels
df = df[df["label"].isin([0, 1])]

print("Total records before duplicate removal:", len(df))


# 5. Detect conflicting labels
print("\nChecking conflicting labels...")

label_counts = df.groupby("URL")["label"].nunique()

conflicting_urls = label_counts[label_counts > 1].index

print("URLs with conflicting labels:", len(conflicting_urls))

# Remove URLs with conflicting labels
df = df[~df["URL"].isin(conflicting_urls)]


# 6. Remove duplicate URLs
df = (
    df.groupby("URL", as_index=False)
    .agg(
        label=("label", "first"),
        source=("source", lambda x: ", ".join(sorted(set(x))))
    )
)

# 7. Display final results
print("\n========== FINAL DATASET ==========")

print("Total unique URLs:", len(df))

print("\nLabel distribution (0=Phishing, 1=Legitimate):")
print(df["label"].value_counts().sort_index())

print("\nSource distribution:")
print(df["source"].value_counts())

# 8. Save new dataset
df.to_csv("dataset/combined_urls.csv", index=False)

print("\nCombined dataset saved successfully!")
print("Location: dataset/combined_urls.csv")
