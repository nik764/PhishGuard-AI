
import pandas as pd
from pathlib import Path
from math import isclose
from url_features import extract_features

# Locate and load dataset
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"

csv_files = list(dataset_folder.rglob("*.csv"))
df = pd.read_csv(csv_files[0])

# Features to compare
features = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]

# Compare the first 3 dataset records
for index in range(3):

    original_url = str(df.loc[index, "URL"])
    extracted = extract_features(original_url)

    print("\n===================================")
    print("DATASET RECORD:", index)
    print("URL:", original_url)
    print("===================================")

    for feature in features:

        original_value = df.loc[index, feature]
        extracted_value = extracted[feature]

        if isclose(
            float(original_value),
            float(extracted_value),
            rel_tol=1e-5,
            abs_tol=1e-5
        ):
            result = "MATCH"
        else:
            result = "DIFFERENT"

        print(
            f"{feature}: Dataset={original_value}, "
            f"Our={extracted_value} -> {result}"
        )

print("\nComparison completed!")
