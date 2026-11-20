
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Locate project folder
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "phishing_model.pkl"

# Load trained model
model = joblib.load(model_path)

# Get feature importance
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

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

# Sort from highest to lowest
importance = importance.sort_values(
    by="Importance",
    ascending=True
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance.sort_values(
    by="Importance",
    ascending=False
).to_string(index=False))

# Create graph
plt.figure(figsize=(10, 7))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("URL Features")

plt.tight_layout()

# Save graph
plt.savefig(project_folder / "feature_importance.png")

plt.show()

print("\nFeature importance analysis completed!")
