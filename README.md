# ReviewLens — IMDB Sentiment Analyzer

ReviewLens reads a movie review and tells you whether it's Positive or
Negative. It's a small web app built on top of a real trained machine
learning model — every prediction you see comes from that model, not
from a guess or a random result.

## What it does

- You type or paste a movie review into the browser.
- The review is sent to a local server running your trained model.
- The model cleans the text, converts it into numbers, and predicts
  a sentiment.
- The result (Positive / Negative + confidence %) is shown instantly.

## How the model works

The model follows four steps:

1. **Preprocessing** — the review is lowercased, HTML tags are removed,
   and anything that isn't a letter or a space is stripped out.
2. **TF-IDF** — the cleaned text is turned into numbers. Common words
   ("the", "was", "it") get a low score. Rare, distinctive words
   ("boring", "brilliant") get a higher score.
3. **Logistic Regression** — a model trained on 50,000 IMDB reviews
   uses those scores to calculate how likely the review is to be
   positive.
4. **Prediction** — the result is converted into a class (Positive or
   Negative) along with a confidence percentage.

This is the same pipeline used to train the model — nothing about it
changes between training and the live app.

## Project structure

```
ReviewLens/
├── README.md
├── IMDB-Dataset.csv       (training data)
├── train_model.py         (trains the model)
├── app.py                 (serves live predictions)
├── model.pkl              (trained model, created by train_model.py)
├── vectorizer.pkl         (fitted TF-IDF vectorizer)
├── requirements.txt
├── index.html
├── style.css
└── script.js
```

## Requirements

- Python 3.9 or newer
- pip

## Setup

### 1. Install backend dependencies

Open a terminal and run:

```bash
pip install -r requirements.txt
```

### 2. Train the model

This step reads `IMDB-Dataset.csv`, trains the Logistic Regression
model, and saves it as `model.pkl` and `vectorizer.pkl`. You only
need to do this once — if those two files already exist, you can
skip this step.

```bash
python train_model.py
```

You should see output like:

```
Accuracy: 0.8925
Saved model.pkl and vectorizer.pkl
```

### 3. Start the backend server

Run:

```bash
python app.py
```

This starts a local server at `http://127.0.0.1:5000` with a
`/predict` endpoint. **Keep this terminal open** — the frontend needs
it running to get real predictions.

### 4. Open the frontend

Open `index.html` in your browser.

## Using the app

1. Go to the **Live Sentiment Analysis** section.
2. Type or paste any review, or click one of the **Sample Reviews**.
3. Click **Analyze Sentiment**.
4. The result card shows Positive or Negative, with a confidence
   percentage.
5. Repeat as many times as you like — there's no limit.

## Troubleshooting

**"Could not reach the model server"**
The backend (`app.py`) isn't running. Go back to the first terminal
and make sure it's still active, or restart it with `python app.py`.

**`model.pkl` or `vectorizer.pkl` not found**
Run `python train_model.py` first — these
files are created by that script and aren't included until you do.

**Port already in use**
Something else is already using port 5000. Close that
program, or change the port number in `app.py`.

## Model details

| | |
|---|---|
| Dataset | IMDB Reviews (50,000 labeled reviews) |
| Model | Logistic Regression |
| Features | TF-IDF (max 5,000 features) |
| Classes | Positive / Negative |
| Accuracy | ~89% on held-out test data |
