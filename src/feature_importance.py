
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Locate project folder
project_folder = Path(__file__).resolve().parent.parent

# Load weighted model without HTTPS
model_path = (
    project_folder
    / "models"
    / "phishing_url_model_weighted_no_https.pkl"
)

model = joblib.load(model_path)

# Get feature names and importance scores
features = model.feature_names_in_

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

# Sort from highest to lowest
importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE (WITHOUT HTTPS) ==========")
print(importance.to_string(index=False))

# Create graph
plt.figure(figsize=(11, 8))

plt.barh(
    importance["Feature"][::-1],
    importance["Importance"][::-1]
)

plt.title("PhishGuard AI - Feature Importance (Without HTTPS)")
plt.xlabel("Importance Score")
plt.ylabel("URL Features")

plt.tight_layout()

# Save graph separately
graph_path = project_folder / "feature_importance_no_https.png"

plt.savefig(graph_path, dpi=300)

plt.show()

print("\nFeature importance analysis completed!")
print("Graph saved at:", graph_path)
