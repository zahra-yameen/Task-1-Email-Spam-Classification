# Task-1-Email-Spam-Classification
Email spam classification app for Arch Technologies

# Email Spam Classification Using Machine Learning

## Overview

A machine learning project that classifies text messages as spam or not spam using TF-IDF feature extraction and a Multinomial Naive Bayes classifier.

## Technologies

- Python
- pandas
- scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Matplotlib and Seaborn
- Joblib
- Streamlit

## Workflow

1. Load the labelled dataset.
2. Clean missing values, labels and duplicates.
3. Split the data into training and testing sets.
4. Extract TF-IDF features.
5. Train the classifier.
6. Evaluate accuracy, precision, recall, F1-score and the confusion matrix.
7. Save the fitted pipeline.
8. Run the Streamlit application.

## Project Structure

- `train_model.py` — training and evaluation
- `app.py` — Streamlit application
- `dataset/` — training data, if redistribution is permitted
- `models/` — saved model pipeline
- `results/` — evaluation charts
- `requirements.txt` — dependencies

## Run Locally

Install dependencies:

`pip install -r requirements.txt`

Train the model:

`python train_model.py`

Run the app:

`python -m streamlit run app.py`

## Evaluation Results

- Accuracy: [Enter actual result]
- Spam precision: [Enter actual result]
- Spam recall: [Enter actual result]
- Spam F1-score: [Enter actual result]

See `results/confusion_matrix.png` for the confusion matrix.

## Limitations

Predictions depend on the training dataset and the patterns learned by the model. The classification should not be treated as a guarantee of message safety.

## Author

Zahra Yameen

# Portfolio Link:

https://task-1-email-spam-classification-g3ifarq4ptoqvjwxs4nmbp.streamlit.app/
