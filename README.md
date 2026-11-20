
# 🛡️ PhishGuard AI
### AI-Powered Phishing URL Detection System

PhishGuard AI is a machine learning-based web application that analyzes URLs and predicts whether they are legitimate or potentially phishing websites.

The project uses a Random Forest classifier trained on URL-based features and provides an interactive dashboard built with Streamlit.

## 🚀 Features

- Detects potentially phishing URLs.
- Classifies URLs as Legitimate or Potential Phishing.
- Extracts 18 URL-based features.
- Displays prediction confidence.
- Maintains scan history during the active session.
- Provides an interactive Streamlit dashboard.
- Displays extracted URL features.

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

## 🤖 Machine Learning Model

- Algorithm: Random Forest Classifier
- Number of estimators: 100
- Dataset: PhiUSIIL Phishing URL Dataset
- Test accuracy: 99.47%

The accuracy is measured on a held-out test split of the dataset. It is not a guarantee of real-world detection performance.

## 📂 Project Structure

```text
PhishGuard-AI/
├── models/
│   └── phishing_url_model.pkl
├── src/
│   ├── url_features.py
│   └── other project scripts
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd PhishGuard-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run main.py
```

## 📊 Dataset

This project uses the PhiUSIIL Phishing URL Dataset from the UCI Machine Learning Repository.

Dataset:
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

The dataset is not included in this repository.

## ⚠️ Disclaimer

PhishGuard AI is an educational machine learning project. Predictions are based on URL characteristics and may be incorrect. Always verify suspicious websites through trusted security tools.

## 👨‍💻 Project

**PhishGuard AI – AI-Powered Phishing Detection System**
