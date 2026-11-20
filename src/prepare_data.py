
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Locate project folder
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"

# Load dataset
csv_files = list(dataset_folder.rglob("*.csv"))
df = pd.read_csv(csv_files[0])

# Select our 18 features
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

# Input features and target label
X = df[selected_features]
y = df["label"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display results
print("\n========== DATA SPLIT RESULTS ==========")

print("Original dataset:", X.shape)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

print("\nData preparation completed!")
