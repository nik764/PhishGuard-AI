
import pandas as pd
import joblib

from url_features import extract_features


# Load our newly trained model
model = joblib.load("models/phishing_url_model.pkl")

print("\n========== PHISHING URL DETECTOR ==========")

# Ask the user for a URL
url = input("\nEnter a URL to check: ")

# Extract the 18 features
features = extract_features(url)

# Convert features into a DataFrame
input_data = pd.DataFrame([features])

# Make prediction
prediction = model.predict(input_data)[0]

# Get prediction confidence
probabilities = model.predict_proba(input_data)[0]
confidence = probabilities[list(model.classes_).index(prediction)]

print("\n========== SCAN RESULT ==========")

if prediction == 0:
    print("Result: PHISHING URL")
else:
    print("Result: LEGITIMATE URL")

print("Confidence:", round(confidence * 100, 2), "%")

print("\nScan completed!")
