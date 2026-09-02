"""
Simple Flask API that loads YOUR trained model (model.pkl + vectorizer.pkl)
and serves live predictions to the frontend.

No new model is created here - this only loads and uses the model
that train_model.py produced from your notebook's exact pipeline.
"""

import os
import re
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows the HTML/JS frontend to call this API

# ---- Load the trained model + vectorizer (created by train_model.py) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ---- Same preprocessing function used during training ----
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    if os.path.exists(os.path.join(BASE_DIR, path)):
        return send_from_directory(BASE_DIR, path)
    return "Not Found", 404


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
@app.route("/api/index", methods=["POST"])
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
    confidence = round(float(max(probabilities)) * 100, 2)

    return jsonify({
        "sentiment": str(prediction).upper(),
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
