// Address of your local Flask backend (see app.py)
const API_URL = "http://127.0.0.1:5000/predict";

const reviewInput = document.getElementById("review-input");
const analyzeBtn = document.getElementById("analyze-btn");
const resultCard = document.getElementById("result-card");
const resultLabel = document.getElementById("result-label");
const resultConfidence = document.getElementById("result-confidence");
const errorMessage = document.getElementById("error-message");

async function analyzeSentiment() {
  const review = reviewInput.value.trim();

  errorMessage.classList.add("hidden");

  if (!review) {
    errorMessage.textContent = "Please write a review before analyzing.";
    errorMessage.classList.remove("hidden");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review })
    });

    if (!response.ok) {
      throw new Error("The model server returned an error.");
    }

    const data = await response.json();
    showResult(data);
  } catch (err) {
    errorMessage.textContent =
      "Could not reach the model server. Make sure app.py is running on port 5000.";
    errorMessage.classList.remove("hidden");
    resultCard.classList.add("hidden");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Sentiment";
  }
}

function showResult(data) {
  const isPositive = data.sentiment === "POSITIVE";

  resultLabel.textContent = isPositive ? "✓ Positive" : "✗ Negative";
  resultLabel.className = "result-label " + (isPositive ? "positive" : "negative");

  // Only show confidence if the backend actually provided it
  if (typeof data.confidence === "number") {
    resultConfidence.textContent = `Confidence: ${data.confidence}%`;
  } else {
    resultConfidence.textContent = "";
  }

  resultCard.classList.remove("hidden");
}

analyzeBtn.addEventListener("click", analyzeSentiment);

// Allow Ctrl+Enter / Cmd+Enter to trigger analysis from the textarea
reviewInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    analyzeSentiment();
  }
});

// Sample review cards load their text into the textarea and analyze it
document.querySelectorAll(".sample-card").forEach((card) => {
  card.addEventListener("click", () => {
    reviewInput.value = card.dataset.review;
    reviewInput.scrollIntoView({ behavior: "smooth", block: "center" });
    analyzeSentiment();
  });
});

/* =====================================================================
   HOW IT WORKS — interactive pipeline inspector
   Step 1 runs the REAL preprocessing regex your model uses.
   Steps 2-3 are clearly-labeled conceptual illustrations, since the
   actual TF-IDF weights and Logistic Regression coefficients only exist
   inside your trained vectorizer.pkl / model.pkl on the backend.
   Step 4 hands off to the real model via the Live Analysis section.
   ===================================================================== */

const pipelineNodes = document.querySelectorAll(".pipeline-node");
const demoInput = document.getElementById("demo-input");
const stepDetail = document.getElementById("step-detail");
const xrayContent = document.getElementById("xray-content");
const theoryCard = document.getElementById("theory-card");

let currentStep = 1;

// Same cleaning rules as app.py's preprocess_text() — nothing invented
function realPreprocess(text) {
  let cleaned = text.toLowerCase();
  cleaned = cleaned.replace(/<[^>]+>/g, "");
  cleaned = cleaned.replace(/[^a-z\s]/g, "");
  return cleaned.replace(/\s+/g, " ").trim();
}

// A small reference list used only to illustrate "common vs. distinctive"
// words for the TF-IDF explainer. This is not your model's vocabulary.
const COMMON_WORDS = new Set([
  "a","an","the","is","was","were","it","this","that","and","but","of",
  "to","in","on","for","with","i","you","he","she","they","we","be",
  "so","very","really","not","no"
]);

function renderStep() {
  if (!stepDetail || !xrayContent || !theoryCard) return; // safety guard

  const raw = demoInput ? demoInput.value : "";
  const cleaned = realPreprocess(raw);
  const words = cleaned.split(" ").filter(Boolean);

  if (currentStep === 1) {
    stepDetail.innerHTML = `
      <h3>Text Sanitization</h3>
      <p>Runs live, using the exact same rules as the trained model:
      lowercase the text, strip HTML tags, then remove anything that
      isn't a letter or a space.</p>
    `;
    xrayContent.innerHTML = `
      <div><span style="color:var(--muted)">input →</span> ${escapeHtml(raw)}</div>
      <div style="margin-top:10px"><span style="color:var(--gold)">cleaned →</span> ${escapeHtml(cleaned) || "<em>empty</em>"}</div>
    `;
    theoryCard.innerHTML = `
      <strong>In simple words:</strong> the model can only read plain words,
      not messy text. So before it looks at anything, we tidy the review up —
      no capital letters, no HTML, no punctuation. That way "Great!" and
      "great" are treated as the exact same word.
    `;
  }

  if (currentStep === 2) {
    stepDetail.innerHTML = `
      <h3>TF-IDF Vectorization</h3>
      <p>Each remaining word is scored by how often it appears in this
      review versus how common it is across all 50,000 training reviews.
      Common words score low; distinctive words score high.</p>
    `;
    if (words.length === 0) {
      xrayContent.innerHTML = `<em style="color:var(--muted)">Type something above to see it tokenized.</em>`;
    } else {
      xrayContent.innerHTML = words
        .map((w) => {
          const isCommon = COMMON_WORDS.has(w);
          return `<span class="token-chip ${isCommon ? "weak" : "strong"}">${escapeHtml(w)}</span>`;
        })
        .join("");
    }
    theoryCard.innerHTML = `
      <strong>In simple words:</strong> some words show up in almost every
      review ("the", "was", "it") and don't tell us much. Other words are
      rarer and much more telling ("boring", "brilliant"). TF-IDF is just a
      way of giving the telling words more importance and the everyday
      words less.
      <span class="disclaimer">The highlighting above is a simplified example — your model's real word scores are stored inside vectorizer.pkl.</span>
    `;
  }

  if (currentStep === 3) {
    const strongCount = words.filter((w) => !COMMON_WORDS.has(w)).length;
    stepDetail.innerHTML = `
      <h3>Logistic Decision</h3>
      <p>The TF-IDF scores are combined into a weighted sum. Logistic
      Regression passes that sum through a sigmoid function, squashing it
      into a value between 0 and 1.</p>
    `;
    xrayContent.innerHTML = `
      <div>weighted sum of ${strongCount} distinctive word${strongCount === 1 ? "" : "s"}</div>
      <div style="margin-top:8px;color:var(--gold)">↓ sigmoid(x) = 1 / (1 + e⁻ˣ)</div>
      <div style="margin-top:8px;color:var(--muted)">output range: 0.0 → 1.0</div>
    `;
    theoryCard.innerHTML = `
      <strong>In simple words:</strong> imagine every important word gets a
      small "vote" — positive words push the score up, negative words pull
      it down. The model adds up all the votes, then squeezes the total
      into a number between 0 and 1 using a formula called sigmoid.
      <span class="disclaimer">The exact vote sizes for each word were learned during training and are saved inside model.pkl.</span>
    `;
  }

  if (currentStep === 4) {
    stepDetail.innerHTML = `
      <h3>Confidence Output</h3>
      <p>If the sigmoid output is above 0.5, the model calls it Positive;
      otherwise, Negative. The distance from 0.5 becomes the confidence
      percentage — <strong>predict_proba()</strong> in scikit-learn.</p>
      <button type="button" class="run-real-model" id="run-real-model-btn">
        Run this exact text through the real model ↓
      </button>
    `;
    xrayContent.innerHTML = `
      <div>0.0 ─────────── 0.5 ─────────── 1.0</div>
      <div style="margin-top:6px;color:var(--muted)">Negative &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; Positive</div>
    `;
    theoryCard.innerHTML = `
      <strong>In simple words:</strong> above 0.5 means "Positive," below
      0.5 means "Negative." The further the number is from 0.5, the more
      confident the model is. Click the button below to see this happen
      for real, with your exact text and your actual trained model.
    `;

    const runBtn = document.getElementById("run-real-model-btn");
    if (runBtn) {
      runBtn.addEventListener("click", () => {
        reviewInput.value = raw;
        document.getElementById("analyze").scrollIntoView({ behavior: "smooth", block: "start" });
        analyzeSentiment();
      });
    }
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

pipelineNodes.forEach((node) => {
  node.addEventListener("click", () => {
    pipelineNodes.forEach((n) => n.classList.remove("active"));
    node.classList.add("active");
    currentStep = Number(node.dataset.step);
    renderStep();
  });
});

if (demoInput) {
  demoInput.addEventListener("input", renderStep);
}

renderStep();
