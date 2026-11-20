
from url_features import extract_features

# Test URL
test_url = "https://www.google.com"

# Extract features
features = extract_features(test_url)

print("\n========== URL FEATURE EXTRACTION ==========")

print("Test URL:", test_url)

print("\nExtracted Features:")

for feature, value in features.items():
    print(f"{feature}: {value}")

print("\nTotal features extracted:", len(features))
