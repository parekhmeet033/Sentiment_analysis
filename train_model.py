"""
Trains the sentiment analysis model using the EXACT same steps as the
original notebook (Sentiment Analysis.ipynb):

    Preprocessing -> TF-IDF -> Logistic Regression

Run this once to create model.pkl and vectorizer.pkl.
These files are then loaded by app.py to serve live predictions.
"""

import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---- STEP 1: Load dataset (same file used in the notebook) ----
df = pd.read_csv("IMDB-Dataset.csv")

X = df["review"]
y = df["sentiment"]


# ---- STEP 2: Preprocessing (identical to the notebook) ----
def preprocess_text(text):
    text = text.lower()                     # lowercase
    text = re.sub(r'<[^>]+>', '', text)      # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)     # remove special characters
    return text


X = X.apply(preprocess_text)

# ---- STEP 3: Train / test split (same params as the notebook) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- STEP 4: TF-IDF vectorization (same params as the notebook) ----
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---- STEP 5: Logistic Regression model (same params as the notebook) ----
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ---- STEP 6: Evaluate (same as the notebook) ----
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ---- STEP 7: Save the trained model + vectorizer for the web app ----
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved model.pkl and vectorizer.pkl")
