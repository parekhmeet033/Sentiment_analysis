"""
Simple Flask API that loads YOUR trained model (model.pkl + vectorizer.pkl)
and serves live predictions to the frontend.

No new model is created here - this only loads and uses the model
that train_model.py produced from your notebook's exact pipeline.
"""

import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows the HTML/JS frontend to call this API

# ---- Load the trained model + vectorizer (created by train_model.py) ----
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# ---- Same preprocessing function used during training ----
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    review = data.get("review", "").strip()

    if not review:
        return jsonify({"error": "Please enter a review."}), 400

    # Preprocess exactly like training data
    cleaned = preprocess_text(review)

    # Vectorize using the SAME fitted TF-IDF vectorizer
    features = vectorizer.transform([cleaned])

    # Predict using the trained Logistic Regression model
    prediction = model.predict(features)[0]

    # Confidence, since LogisticRegression supports predict_proba
    probabilities = model.predict_proba(features)[0]
    confidence = round(max(probabilities) * 100, 2)

    return jsonify({
        "sentiment": str(prediction).upper(),
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
