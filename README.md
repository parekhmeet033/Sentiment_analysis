# ReviewLens

**Live site:** https://sentiment-analysis-nu-cyan.vercel.app/
**Source code:** https://github.com/parekhmeet033/Sentiment_analysis

ReviewLens is a movie review sentiment analyzer. You give it a review —
your own words, not a pre-written example — and it tells you whether
the review reads as Positive or Negative, along with how confident it
is in that call. Every result comes from a real trained machine
learning model running on the server; nothing on the results screen is
mocked, randomized, or hardcoded.

## What problem it solves

Reading sentiment out of free-form text is something people do
instinctively but computers have to learn. ReviewLens is a working,
end-to-end example of that: raw English text goes in, a probability
comes out, and the whole path between those two points — cleaning the
text, turning it into numbers, and scoring it — is visible and
explained inside the app itself, not just in this README.

## How it works

The model follows a four-stage pipeline, and the site's **How It
Works** section lets you step through each stage interactively on
your own text before running it for real:

1. **Preprocessing** — the review is lowercased, HTML tags are
   stripped, and anything that isn't a letter or a space is removed.
   This keeps the model from treating "Great!" and "great" as
   different words.
2. **TF-IDF Vectorization** — the cleaned text is converted into
   numbers. Words that appear in almost every review ("the", "was",
   "it") get a low score; rare, opinion-carrying words ("boring",
   "brilliant") get a higher one. This is what lets a model built on
   plain word counts pick up on tone at all.
3. **Logistic Regression** — a model trained on 50,000 labeled IMDB
   reviews combines those word scores into a single weighted sum, then
   passes it through a sigmoid function to squeeze it into a value
   between 0 and 1.
4. **Prediction** — a score above 0.5 is called Positive, below 0.5 is
   called Negative. How far the score sits from 0.5 becomes the
   confidence percentage shown in the result.

## Features

- **Live Sentiment Analysis** — type or paste any review and get an
  instant Positive/Negative call with a confidence score, pulled from
  the real model on every request.
- **Advanced Glassmorphism UI** — High-refinement liquid glass optics with ambient background mesh gradients, ray-traced specular highlights, and fluid backdrop blur effects (`blur(32px)`).
- **Interactive How It Works panel** — click through the four pipeline
  stages and watch your own text get cleaned, tokenized, and scored in
  real time, with plain-language explanations at each step.
- **Sample reviews** — three pre-written reviews to try the model on
  immediately, no typing required.
- **Model information panel** — dataset, algorithm, feature type, and
  class labels, all in one place.
- **No fake results anywhere** — every number on the page is either
  computed from your input on the spot, or clearly labeled as a
  simplified explanation rather than the model's actual internals.

## How it's built

**Frontend:** plain HTML, CSS, and JavaScript with an Advanced Glassmorphism design system — no framework, no build
step. `script.js` handles the live analysis calls, the interactive
pipeline walkthrough, and the sample review cards.

**Backend:** a Flask app (`api/index.py`) exposing a `/predict`
endpoint, deployed as a serverless function on Vercel. It loads the
already-trained model and vectorizer on startup and serves both the
static site and the prediction API from a single deployment.

**Model:** trained offline with `train_model.py` on the IMDB Reviews
dataset (50,000 labeled reviews, evenly split between positive and
negative). Training uses a TF-IDF vectorizer capped at 5,000 features
and a scikit-learn Logistic Regression classifier, split 80/20 for
train and test. The trained model reaches roughly 89% accuracy on the
held-out test set. The fitted model and vectorizer are saved with
`joblib` (`model.pkl`, `vectorizer.pkl`) and loaded directly by the
live API — the same objects used during training are what serve every
request in production, nothing is retrained or approximated at
request time.

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Flask, Flask-CORS |
| Model | scikit-learn (TF-IDF + Logistic Regression) |
| Dataset | IMDB Reviews (50,000 rows) |
| Hosting | Vercel (serverless Python function) |
