import streamlit as st
import joblib
import numpy as np
import re
import os
from tensorflow.keras.models import load_model
from textblob import TextBlob
from dotenv import load_dotenv
from groq import Groq

import warnings
warnings.filterwarnings("ignore")
# =========================
# CONFIG
# =========================
load_dotenv()

st.set_page_config(page_title="Fake News Detector", layout="wide")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_TEXT_LEN = 2000
SEQ_LEN = 30


# =========================
# TEXT CLEANING
# =========================
def clean_text_basic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text_strict(text: str) -> str:
    text = re.sub(r"http\S+", " ", str(text or ""))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# =========================
# SENTIMENT
# =========================
def get_sentiment(text: str):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive 😊", polarity
    elif polarity < -0.1:
        return "Negative 😡", polarity
    return "Neutral 😐", polarity


# =========================
# LLM FACT CHECK
# =========================
def verify_with_llm(text: str):
    try:
        text = text[:MAX_TEXT_LEN]

        prompt = f"""
You are a strict fact-checking system.

Return ONLY in this format:

VERDICT: REAL / FAKE / UNVERIFIED
REASON: short explanation
EVIDENCE: low / medium / high

News:
\"\"\"{text}\"\"\"
"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {str(e)}"

def parse_llm_output(llm_text: str):
    text = llm_text.upper()

    verdict = "UNVERIFIED"
    confidence = "low"

    if "VERDICT: REAL" in text:
        verdict = "REAL"
    elif "VERDICT: FAKE" in text:
        verdict = "FAKE"

    if "EVIDENCE: HIGH" in text:
        confidence = "high"
    elif "EVIDENCE: MEDIUM" in text:
        confidence = "medium"

    return verdict, confidence
# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_models():
    try:
        tfidf_d1 = joblib.load("d1_tfidf.pkl")
        clf_d1 = joblib.load("d1_svm.pkl")

        tfidf_d3 = joblib.load("d3_tfidf.pkl")
        clf_d3 = joblib.load("d3_gbm.pkl")

        try:
            model_d2 = load_model("d2_bilstm.h5")
            vocab_d2 = joblib.load("d2_vocab.pkl")
        except Exception:
            model_d2, vocab_d2 = None, None

        return {
            "d1": (tfidf_d1, clf_d1),
            "d2": (model_d2, vocab_d2),
            "d3": (tfidf_d3, clf_d3)
        }

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


models = load_models()


# =========================
# PREDICTION HELPERS
# =========================
def predict_d1(text, tfidf, clf):
    vec = tfidf.transform([clean_text_basic(text)])
    p = clf.predict_proba(vec)[0]
    return p


def predict_d2(text, model, vocab):
    if not model or not vocab:
        return None

    seq = [vocab.get(w, 1) for w in text.lower().split()]
    seq = seq[:SEQ_LEN] + [0] * max(0, SEQ_LEN - len(seq))

    return float(model.predict(np.array([seq]), verbose=0)[0][0])


def predict_d3(text, tfidf, clf):
    vec = tfidf.transform([clean_text_strict(text)])
    p = clf.predict_proba(vec)[0]
    return p


# =========================
# ENSEMBLE
# =========================
def ensemble(p1, p2, p3):
    probs = []
    weights = []

    # D1 (SVM)
    w1 = 0.2 if (p1[0] > 0.95 or p1[0] < 0.05) else 0.4
    probs.append(p1[0])
    weights.append(w1)

    # D2 (BiLSTM)
    if p2 is not None:
        probs.append(p2)
        weights.append(0.3)

    # D3 (GBM)
    probs.append(p3[0])
    weights.append(0.5)

    avg_fake = np.average(probs, weights=weights)
    spread = max(probs) - min(probs)

    return avg_fake, spread


# =========================
# UI
# =========================
st.title("🧠 Fake News Detection System")

option = st.radio("Input Type", ["Text", "File"])
text = ""

if option == "Text":
    text = st.text_area("Enter news text")

elif option == "File":
    file = st.file_uploader("Upload .txt file")
    if file:
        text = file.read().decode("utf-8")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    if not models:
        st.error("Models not loaded")
        st.stop()

    (tfidf_d1, clf_d1) = models["d1"]
    (model_d2, vocab_d2) = models["d2"]
    (tfidf_d3, clf_d3) = models["d3"]

    # Predictions
    p1 = predict_d1(text, tfidf_d1, clf_d1)
    p2 = predict_d2(text, model_d2, vocab_d2)
    p3 = predict_d3(text, tfidf_d3, clf_d3)

    avg_fake, spread = ensemble(p1, p2, p3)

    # Core metrics (defined ONCE)
    confidence = 1 - spread
    uncertainty_flag = spread > 0.6

    # Base ML verdict (MOVED BEFORE usage ✅)
    if avg_fake > 0.6:
        verdict = "🚨 FAKE"
    elif avg_fake < 0.4:
        verdict = "✅ REAL"
    else:
        verdict = "⚠️ UNCERTAIN"

    # LLM
    llm_verdict = None
    llm_confidence = None

    if uncertainty_flag or confidence < 0.6:
        with st.spinner("Running AI fact-check (LLaMA 3.3)..."):
            llm_raw = verify_with_llm(text)
            llm_verdict, llm_confidence = parse_llm_output(llm_raw)

    # Final decision (FIXED order)
    if not uncertainty_flag and confidence > 0.7:
        final_verdict = verdict
    elif llm_confidence == "high":
        if llm_verdict == "REAL":
            final_verdict = "✅ REAL (LLM Verified)"
        elif llm_verdict == "FAKE":
            final_verdict = "🚨 FAKE (LLM Verified)"
        else:
            final_verdict = "⚠️ UNVERIFIED"
    else:
        final_verdict = "⚠️ UNVERIFIED (Low confidence)"

    # Extra info
    risk = (
        "High Risk 🚨" if avg_fake > 0.75
        else "Moderate Risk ⚠️" if avg_fake > 0.55
        else "Low Risk ✅"
    )

    sentiment_label, sentiment_score = get_sentiment(text)

    # Reasons
    reasons = []
    if p1[0] > 0.8:
        reasons.append("SVM strongly indicates FAKE pattern")
    if p3[1] > 0.7:
        reasons.append("GBM strongly indicates REAL pattern")
    if p2 is not None and p2 < 0.3:
        reasons.append("BiLSTM suggests REAL linguistic structure")
    if uncertainty_flag:
        reasons.append("High disagreement between models")
    if not reasons:
        reasons.append("Mixed signals across models")

    # =========================
    # OUTPUT (CLEANED)
    # =========================
    st.subheader("📊 Results")
    c0, c1, c2, c3 = st.columns([1.5, 1, 1, 1])

    with c0:
        st.write("")  # empty space

    with c1:
        st.markdown("**Dataset-1**")

    with c2:
        st.markdown("**Dataset-2**")

    with c3:
        st.markdown("**Dataset-3**")

    # Row 2: label + values
    c0, c1, c2, c3 = st.columns([1.5, 1, 1, 1])

    with c0:
        st.markdown("### Fake Probability")

    with c1:
        st.metric("news.csv", f"{p1[0]*100:.2f}%")

    with c2:
        st.metric(
            "india-news-headlines.csv",
            f"{p2*100:.2f}%" if p2 is not None else "N/A"
        )

    with c3:
        st.metric("data.csv", f"{p3[0]*100:.2f}%")

    st.success(f"Final Verdict: {final_verdict}")

    if uncertainty_flag:
        st.warning("High disagreement between models")

    st.progress(float(confidence))
    st.write(f"Confidence: {confidence * 100:.1f}%")
    st.write(f"Risk Level: {risk}")
    st.write(f"Sentiment: {sentiment_label} ({sentiment_score:.2f})")

    st.write("### 🧠 Explanation")
    for r in reasons:
        st.write(f"- {r}")

    # LLM output (ONLY ONCE)
    if llm_verdict is not None:
        st.write("### 🤖 LLM Fact Check")
        st.write(llm_raw)