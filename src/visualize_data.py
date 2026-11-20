
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate project folder
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"

csv_files = list(dataset_folder.rglob("*.csv"))
df = pd.read_csv(csv_files[0])

# Count labels
label_counts = df["label"].value_counts().sort_index()

# Create chart
plt.figure(figsize=(8, 5))

plt.bar(
    ["Phishing", "Legitimate"],
    [label_counts[0], label_counts[1]]
)

plt.title("Phishing vs Legitimate URLs")
plt.xlabel("Website Type")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.savefig(project_folder / "label_distribution.png")

plt.show()

print("Graph created successfully!")
