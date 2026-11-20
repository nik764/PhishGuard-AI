
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Locate project folder
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"
models_folder = project_folder / "models"

# Create models folder if needed
models_folder.mkdir(exist_ok=True)

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

# Separate input and output
X = df[selected_features]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Save trained model
model_path = models_folder / "phishing_model.pkl"
joblib.dump(model, model_path)

print("\nModel training completed!")
print("Model saved at:", model_path)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
