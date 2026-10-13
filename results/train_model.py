from text_preprocessing import clean_text
import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "spam.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "spam_pipeline.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE, encoding="latin-1")

# Remove common empty extra columns from Kaggle CSV files.
df = df.loc[
    :,
    ~df.columns.astype(str).str.match(r"^Unnamed")
]

# Normalize column names for easier matching.
df.columns = [
    str(column).strip().lower()
    for column in df.columns
]

label_candidates = [
    "label", "labels", "v1", "category", "class", "target"
]

text_candidates = [
    "message", "text", "email", "email_text",
    "body", "content", "v2"
]

label_col = next(
    (column for column in label_candidates if column in df.columns),
    None,
)

text_col = next(
    (column for column in text_candidates if column in df.columns),
    None,
)

if label_col is None or text_col is None:
    raise ValueError(
        f"Could not identify the label/text columns. "
        f"Available columns: {df.columns.tolist()}. "
        f"Update label_candidates and text_candidates "
        f"to match your CSV."
    )

if label_col == text_col:
    raise ValueError("Label and text columns must be different.")

df = df[[label_col, text_col]].copy()
df.columns = ["label", "message"]

print("Original usable shape:", df.shape)
print(df.head())


# --------------------------------------------------
# 3. Clean the dataset
# --------------------------------------------------

df = df.dropna(subset=["label", "message"]).copy()

df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()

df = df[df["message"] != ""]
df = df[df["label"] != ""]

# Standard labels found in common spam datasets.
label_mapping = {
    "spam": 1,
    "ham": 0,
    "not spam": 0,
    "legitimate": 0,
    "legit": 0,
    "real": 0,
    "junk": 1,
    "1": 1,
    "0": 0,
}

df["label"] = df["label"].map(label_mapping)

# Drop labels that were not recognized instead of
# silently assigning the wrong class.
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

df["message"] = df["message"].str.lower()

# Remove duplicate message/label pairs.
df = df.drop_duplicates(subset=["message", "label"])

if df["label"].nunique() != 2:
    raise ValueError(
        "The cleaned dataset must contain both spam (1) "
        "and ham/not-spam (0). Check your label mapping."
    )

print("\nCleaned dataset shape:", df.shape)
print("\nClass counts:")
print(df["label"].value_counts().rename(
    index={0: "Ham / Not Spam", 1: "Spam"}
))


# --------------------------------------------------
# 4. Split data into training and testing sets
# --------------------------------------------------

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Build TF-IDF + Logistic Regression pipeline
# --------------------------------------------------

def clean_text(text):
    """Normalize text without removing useful word signals."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            preprocessor=clean_text,
            stop_words=None,
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight=None,
            random_state=42,
        ),
    ),
])


# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 7. Evaluate on unseen test data
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Ham / Not Spam", "Spam"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )
)


# --------------------------------------------------
# 8. Save the complete pipeline
# --------------------------------------------------

joblib.dump(pipeline, MODEL_FILE)

print("\nModel saved successfully:")
print(MODEL_FILE)


# --------------------------------------------------
# 9. Test with a sample message
# --------------------------------------------------

sample = [
    "Congratulations! You have won a free prize. Claim now!"
]

prediction = pipeline.predict(sample)[0]
probabilities = pipeline.predict_proba(sample)[0]

print("\nSample email:", sample[0])
print("Prediction:", "SPAM" if prediction == 1 else "NOT SPAM")
print("Spam probability:", round(float(probabilities[1]), 4))
