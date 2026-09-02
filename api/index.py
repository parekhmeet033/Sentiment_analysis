import os
import re
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Ensure CORS headers are added to all responses

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Safe loading of model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    load_error = None
except Exception as e:
    model = None
    vectorizer = None
    load_error = str(e)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


@app.route("/", methods=["GET"])
def home():
    if os.path.exists(os.path.join(BASE_DIR, "index.html")):
        return send_from_directory(BASE_DIR, "index.html")
    return jsonify({"status": "API is online", "load_error": load_error})


@app.route("/predict", methods=["GET", "POST"])
@app.route("/api/predict", methods=["GET", "POST"])
@app.route("/api/index", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "status": "API is online",
            "load_error": load_error,
            "model_loaded": model is not None,
            "vectorizer_loaded": vectorizer is not None
        })

    if load_error or model is None or vectorizer is None:
        return jsonify({"error": f"Model initialization failed: {load_error}"}), 500

    data = request.get_json(silent=True) or {}
    review = data.get("review", "").strip()

    if not review:
        return jsonify({"error": "Please enter a review."}), 400

    cleaned = preprocess_text(review)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    return jsonify({
        "sentiment": str(prediction).upper(),
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
