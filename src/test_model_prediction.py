
import pandas as pd
import joblib

from url_features import extract_features


# Load the latest 19-feature model
model = joblib.load("models/phishing_url_model.pkl")

# Test URLs
test_urls = [
    # Popular legitimate websites
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.flipkart.com",
    "https://www.amazon.in",
    "https://www.youtube.com",
    "https://www.wikipedia.org",
    "https://www.github.com",
    "https://www.apple.com",
    "https://www.netflix.com",
    "https://www.linkedin.com",

    # Different URL structures
    "https://www.flipkart.com/",
    "https://www.flipkart.com/search?q=mobile",
    "https://www.amazon.in/s?k=laptop",
    "https://github.com/login",

    # Suspicious-looking test examples (synthetic)
    "http://secure-login-verification-account.xyz/login",
    "http://account-update-confirmation.example/login",
    "http://verify-bank-security.example/account",
    "https://www.djchhduff.com"
]

print("\n========== PHISHGUARD AI - BROAD TESTING ==========")

for url in test_urls:

    features = extract_features(url)
    input_data = pd.DataFrame([features])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities[list(model.classes_).index(prediction)]

    result = "POTENTIAL PHISHING" if prediction == 0 else "LEGITIMATE"

    print("\n----------------------------------------")
    print("URL:", url)
    print("Prediction:", result)
    print("Model confidence:", round(confidence * 100, 2), "%")
    print("Domain entropy:", round(features["DomainNameEntropy"], 4))

print("\n========== TESTING COMPLETED ==========")
