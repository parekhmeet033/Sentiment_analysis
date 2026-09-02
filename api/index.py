import os
import re
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(BASE_DIR, filename)
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
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
