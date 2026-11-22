
import pandas as pd

legit = pd.read_parquet(
    "dataset/Phish360/Phish360_legit.parquet",
    columns=["URL"]
)

phish = pd.read_parquet(
    "dataset/Phish360/Phish360_phish.parquet",
    columns=["URL"]
)

# Remove surrounding whitespace
legit["URL"] = legit["URL"].astype(str).str.strip()
phish["URL"] = phish["URL"].astype(str).str.strip()

# Remove empty URLs
legit = legit[legit["URL"] != ""]
phish = phish[phish["URL"] != ""]

# Check duplicates
print("\n========== DUPLICATE CHECK ==========")

print("Duplicate legitimate URLs:", legit["URL"].duplicated().sum())
print("Duplicate phishing URLs:", phish["URL"].duplicated().sum())

# Check overlap between classes
overlap = set(legit["URL"]) & set(phish["URL"])

print("URLs appearing in BOTH classes:", len(overlap))

if overlap:
    print("\nSample overlapping URLs:")
    print(list(overlap)[:10])

print("\n========== QUALITY CHECK ==========")
print("Clean legitimate URLs:", len(legit))
print("Clean phishing URLs:", len(phish))

print("\nInspection completed!")
