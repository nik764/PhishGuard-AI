
import pandas as pd
from pathlib import Path

# Find the project dataset folder
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"

# Find CSV files, including files inside subfolders
csv_files = list(dataset_folder.rglob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found inside dataset folder.")
    raise SystemExit

print("CSV files found:")

for i, file in enumerate(csv_files, start=1):
    print(f"{i}. {file.name}")

# Load the first CSV file
file_path = csv_files[0]

print("\nLoading dataset...")
df = pd.read_csv(file_path)

# Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nDataset statistics:")
print(df.describe(include="all"))

print("\n========== LABEL DISTRIBUTION ==========")

if "label" in df.columns:
    print(df["label"].value_counts())
elif "Label" in df.columns:
    print(df["Label"].value_counts())
else:
    print("Label column not identified. Check the column names above.")

print("\nDataset analysis completed!")
