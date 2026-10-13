from text_preprocessing import clean_text
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -------------------------------------------
# 1. Page configuration
# -------------------------------------------

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "spam_pipeline.pkl"


# -------------------------------------------
# 2. Load the trained pipeline
# -------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


st.title("📧 Email Spam Detector")

st.write(
    "Enter an email or message to check whether "
    "it is likely to be spam or legitimate."
)

if not MODEL_FILE.exists():
    st.error(
        "The trained model was not found. "
        "Run train_model.py first to create it."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# -------------------------------------------
# 3. Single-message prediction
# -------------------------------------------

st.subheader("Check a single email")

email_text = st.text_area(
    "Email or message text",
    placeholder=(
        "Example: Congratulations! Claim your "
        "free prize now!"
    ),
    height=180,
)

if st.button("Predict", type="primary"):
    if not email_text.strip():
        st.warning("Please enter some text first.")
    else:
        prediction = int(model.predict([email_text])[0])
        probabilities = model.predict_proba([email_text])[0]

        # The training script maps 0 = Ham, 1 = Spam.
        spam_index = list(model.classes_).index(1)
        spam_probability = float(probabilities[spam_index])

        if prediction == 1:
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM (HAM)")

        st.metric(
            "Estimated spam probability",
            f"{spam_probability:.1%}",
        )

        st.caption(
            "This probability is the model's estimate, "
            "not a guarantee that the classification is correct."
        )


# -------------------------------------------
# 4. Batch CSV prediction
# -------------------------------------------

st.divider()
st.subheader("Check multiple messages")

st.write(
    "Upload a CSV file with a column named "
    "'message' or 'text'."
)

uploaded_file = st.file_uploader(
    "Upload a CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        normalized_columns = {
            str(column).strip().lower(): column
            for column in batch_df.columns
        }

        text_column = next(
            (
                normalized_columns[name]
                for name in ["message", "text", "email", "content"]
                if name in normalized_columns
            ),
            None,
        )

        if text_column is None:
            st.error(
                "CSV needs a message, text, email, "
                "or content column."
            )
        else:
            texts = (
                batch_df[text_column]
                .fillna("")
                .astype(str)
                .tolist()
            )

            valid_positions = [
                i for i, text in enumerate(texts)
                if text.strip()
            ]

            results_df = batch_df.copy()
            results_df["prediction"] = "Not processed"
            results_df["spam_probability"] = None

            if valid_positions:
                valid_texts = [
                    texts[i] for i in valid_positions
                ]

                predictions = model.predict(valid_texts)
                probabilities = model.predict_proba(valid_texts)

                spam_index = list(model.classes_).index(1)

                for j, original_index in enumerate(valid_positions):
                    results_df.loc[
                        results_df.index[original_index],
                        "prediction"
                    ] = (
                        "Spam"
                        if int(predictions[j]) == 1
                        else "Not Spam"
                    )

                    results_df.loc[
                        results_df.index[original_index],
                        "spam_probability"
                    ] = float(probabilities[j][spam_index])

            st.dataframe(results_df, use_container_width=True)

            csv_data = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download prediction results",
                data=csv_data,
                file_name="spam_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")


st.divider()
st.caption(
    "Machine learning project using TF-IDF and Logistic Regression."
)
