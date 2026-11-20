
import streamlit as st
import pandas as pd
import joblib

from src.url_features import extract_features


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("models/phishing_url_model.pkl")


model = load_model()


# ---------------- SESSION STATE ----------------

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #e2e8f0;
}

h1, h2, h3 {
    color: #60a5fa;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

st.sidebar.title("🛡️ PhishGuard AI")

st.sidebar.write("AI-Powered Phishing Detection")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Scan History", "About Project"]
)


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("🛡️ PhishGuard AI")
    st.write("AI-Powered Phishing URL Detection System")

    st.markdown("---")

    st.subheader("🔍 Website Security Scanner")

    url = st.text_input(
        "Enter a website URL",
        placeholder="https://www.example.com"
    )

    scan_button = st.button("🚀 Scan URL")

    if scan_button:

        if not url.strip():

            st.warning("Please enter a URL.")

        else:

            try:

                # Extract URL features
                features = extract_features(url)

                # Prepare input
                input_data = pd.DataFrame([features])

                # Model prediction
                prediction = model.predict(input_data)[0]

                probabilities = model.predict_proba(input_data)[0]

                confidence = probabilities[
                    list(model.classes_).index(prediction)
                ]

                # Determine result
                if prediction == 0:

                    result = "Potential Phishing"
                    st.error("⚠️ Potential Phishing URL Detected!")

                else:

                    result = "Legitimate"
                    st.success("✅ URL Classified as Legitimate")

                # Display confidence
                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "This is an automated prediction, not a guarantee "
                    "that the website is safe."
                )

                # Save history
                st.session_state.scan_history.insert(
                    0,
                    {
                        "URL": url,
                        "Result": result,
                        "Confidence": f"{confidence * 100:.2f}%"
                    }
                )

                # Show extracted features
                with st.expander("View Extracted URL Features"):

                    feature_df = pd.DataFrame(
                        features.items(),
                        columns=["Feature", "Value"]
                    )

                    st.dataframe(
                        feature_df,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as e:

                st.error(f"Unable to scan URL: {e}")


# ---------------- SCAN HISTORY ----------------

elif page == "Scan History":

    st.title("📋 Scan History")

    st.write("Previously scanned URLs in this session.")

    if st.session_state.scan_history:

        history_df = pd.DataFrame(
            st.session_state.scan_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Scan History"):

            st.session_state.scan_history = []
            st.rerun()

    else:

        st.info("No URLs have been scanned yet.")


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("ℹ️ About PhishGuard AI")

    st.write("""
    PhishGuard AI is a machine learning-based phishing URL
    detection system developed using Python.

    It analyzes URL characteristics and predicts whether a URL
    is potentially phishing or legitimate.
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest Classifier
    - Streamlit
    - Joblib
    """)

    st.subheader("Machine Learning Model")

    st.write("""
    The system uses a Random Forest Classifier trained on
    18 features extracted directly from URLs.
    """)

    st.subheader("Dataset")

    st.write("""
    PhiUSIIL Phishing URL Dataset
    """)

    st.subheader("Project Purpose")

    st.write("""
    The purpose of this project is to demonstrate how machine
    learning can help identify potentially suspicious URLs.
    """)


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "PhishGuard AI | AI-Powered Phishing Detection System | "
    "Random Forest Classifier"
)
