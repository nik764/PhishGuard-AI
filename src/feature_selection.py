
import pandas as pd
from pathlib import Path

# Locate project folder
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"

# Load dataset
csv_files = list(dataset_folder.rglob("*.csv"))
df = pd.read_csv(csv_files[0])

# Select URL-based features
selected_features = [
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

# Add the target label
selected_features.append("label")

# Create a smaller dataset
selected_df = df[selected_features]

print("\n========== SELECTED FEATURES ==========")
print(selected_df.columns.tolist())

print("\n========== NEW DATASET SHAPE ==========")
print(selected_df.shape)

print("\n========== FIRST 5 RECORDS ==========")
print(selected_df.head())

print("\nFeature selection completed!")
