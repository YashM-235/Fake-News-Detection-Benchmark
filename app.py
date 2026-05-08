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

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_TEXT_LEN = 2000
SEQ_LEN = 30

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #04152d 0%, #071c3b 100%);
    color: white;
}

/* Title */
.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #00d4ff;
    margin-bottom: 20px;
}

/* Text Area */
.stTextArea textarea {
    background-color: #1a2742 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #2b4c7e !important;
    padding: 18px !important;
    font-size: 20px !important;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background-color: #16233c;
    border-radius: 12px;
    padding: 10px;
}

/* Radio */
.stRadio label {
    color: white !important;
}

/* Button */
.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 15px rgba(0,212,255,0.5);
}

/* Model Cards */
.model-card {
    background: rgba(0,255,100,0.08);
    border: 1px solid rgba(0,255,100,0.35);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    min-height: 220px;
    box-shadow: 0 0 15px rgba(0,255,100,0.08);
}

.model-title {
    font-size: 30px;
    color: #8da8c7;
    margin-bottom: 15px;
}

.model-verdict {
    font-size: 36px;
    color: #00ff9c;
    font-weight: bold;
    margin-bottom: 15px;
}

.model-text {
    font-size: 18px;
    color: #d7e3ff;
}

/* Fact Check */
.fact-box {
    margin-top: 20px;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(0,212,255,0.5);
    background: rgba(0,212,255,0.05);
    color: #00d4ff;
    font-size: 22px;
}

/* Summary */
.summary-box {
    margin-top: 14px;
    background: rgba(255,255,255,0.04);
    padding: 18px;
    border-radius: 10px;
    font-size: 22px;
    color: #dce7ff;
}

/* Final Verdict */
.result-box {
    margin-top: 20px;
    padding: 22px;
    border-radius: 14px;
    background: rgba(0,0,0,0.25);
    border-left: 6px solid #00d4ff;
}

.result-title {
    font-size: 34px;
    font-weight: bold;
    color: white;
}

.result-sub {
    font-size: 20px;
    color: #dce7ff;
}

/* Expander */
.streamlit-expanderHeader {
    color: white !important;
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)

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
You are a strict AI fact-checking assistant.

Analyze the following news and return ONLY in this exact format:

VERDICT: REAL / FAKE / UNVERIFIED
REASON: short explanation
EVIDENCE: LOW / MEDIUM / HIGH

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
    confidence = "LOW"

    if "VERDICT: REAL" in text:
        verdict = "REAL"

    elif "VERDICT: FAKE" in text:
        verdict = "FAKE"

    if "EVIDENCE: HIGH" in text:
        confidence = "HIGH"

    elif "EVIDENCE: MEDIUM" in text:
        confidence = "MEDIUM"

    return verdict, confidence


# =========================
# LOAD MODELS
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
# PREDICTIONS
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

    # D1
    w1 = 0.2 if (p1[0] > 0.95 or p1[0] < 0.05) else 0.4
    probs.append(p1[0])
    weights.append(w1)

    # D2
    if p2 is not None:
        probs.append(p2)
        weights.append(0.3)

    # D3
    probs.append(p3[0])
    weights.append(0.5)

    avg_fake = np.average(probs, weights=weights)
    spread = max(probs) - min(probs)

    return avg_fake, spread


# =========================
# HEADER
# =========================
st.markdown(
    '<div class="main-title">🛡️ Fake News Detection System</div>',
    unsafe_allow_html=True
)

# =========================
# INPUT SECTION
# =========================
option = st.radio("Input Type", ["Text", "File"])

text = ""

if option == "Text":

    text = st.text_area(
        "📰 Paste news article or headline below:",
        height=140,
        placeholder="Scientists at MIT confirm new battery technology doubles EV range..."
    )

elif option == "File":

    file = st.file_uploader("Upload .txt file")

    if file:
        text = file.read().decode("utf-8")


# =========================
# ANALYZE BUTTON
# =========================
if st.button("▶ Analyse Now"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    if not models:
        st.error("Models not loaded")
        st.stop()

    (tfidf_d1, clf_d1) = models["d1"]
    (model_d2, vocab_d2) = models["d2"]
    (tfidf_d3, clf_d3) = models["d3"]

    # =========================
    # MODEL PREDICTIONS
    # =========================
    p1 = predict_d1(text, tfidf_d1, clf_d1)
    p2 = predict_d2(text, model_d2, vocab_d2)
    p3 = predict_d3(text, tfidf_d3, clf_d3)

    avg_fake, spread = ensemble(p1, p2, p3)

    confidence = 1 - spread
    uncertainty_flag = spread > 0.6

    # =========================
    # FINAL VERDICT
    # =========================
    if avg_fake > 0.6:
        verdict = "FAKE"
        verdict_icon = "🚨"

    elif avg_fake < 0.4:
        verdict = "REAL"
        verdict_icon = "✅"

    else:
        verdict = "UNCERTAIN"
        verdict_icon = "⚠️"

    # =========================
    # LLM FACT CHECK
    # =========================
    with st.spinner("🤖 AI Fact-Check (LLaMA 70B) running..."):

        llm_raw = verify_with_llm(text)
        llm_verdict, llm_confidence = parse_llm_output(llm_raw)

    # =========================
    # SENTIMENT
    # =========================
    sentiment_label, sentiment_score = get_sentiment(text)

    # =========================
    # MODEL CARDS
    # =========================
    c1, c2, c3 = st.columns(3)

    # D1
    with c1:

        d1_label = "REAL" if p1[0] < 0.5 else "FAKE"

        st.markdown(f"""
        <div class="model-card">
            <div class="model-title">D1 · SVM</div>
            <div class="model-verdict">{d1_label}</div>
            <div class="model-text">
                Confidence: {(max(p1)*100):.1f}%<br>
                Risk: {p1[0]:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # D2
    with c2:

        if p2 is not None:

            d2_label = "REAL" if p2 < 0.5 else "FAKE"

            d2_conf = (1 - abs(0.5 - p2) * 2) * 100

            st.markdown(f"""
            <div class="model-card">
                <div class="model-title">D2 · BiLSTM</div>
                <div class="model-verdict">{d2_label}</div>
                <div class="model-text">
                    Confidence: {d2_conf:.1f}%<br>
                    Risk: {p2:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="model-card">
                <div class="model-title">D2 · BiLSTM</div>
                <div class="model-verdict">N/A</div>
                <div class="model-text">
                    Model unavailable
                </div>
            </div>
            """, unsafe_allow_html=True)

    # D3
    with c3:

        d3_label = "REAL" if p3[0] < 0.5 else "FAKE"

        st.markdown(f"""
        <div class="model-card">
            <div class="model-title">D3 · GBM</div>
            <div class="model-verdict">{d3_label}</div>
            <div class="model-text">
                Confidence: {(max(p3)*100):.1f}%<br>
                Risk: {p3[0]:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # FACT CHECK BOX
    # =========================
    preview = text[:60].replace("\n", " ")

    st.markdown(f"""
    <div class="fact-box">
        🤖 AI Fact-Check (Llama 70B):
        Searching for '<b>{preview}...</b>'
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # SUMMARY
    # =========================
    st.markdown(f"""
    <div class="summary-box">
        ✓ LLM Verdict: <b>{llm_verdict}</b>
        &nbsp;&nbsp; · &nbsp;&nbsp;
        Credibility: <b>{llm_confidence}</b>
        &nbsp;&nbsp; · &nbsp;&nbsp;
        Sentiment: <b>{sentiment_label}</b>
        ({sentiment_score:+.2f})
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # FINAL RESULT
    # =========================
    st.markdown(f"""
    <div class="result-box">
        <div class="result-title">
            {verdict_icon} FINAL VERDICT: {verdict}
        </div>

        <div class="result-sub">
            Overall Confidence: {confidence*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # CONFIDENCE BAR
    # =========================
    st.progress(float(confidence))

    # =========================
    # EXPLANATION
    # =========================
    st.markdown("## 🧠 Model Explanation")

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

    for r in reasons:
        st.write(f"• {r}")

    # =========================
    # RAW LLM OUTPUT
    # =========================
    with st.expander("🔍 Detailed LLM Response"):
        st.code(llm_raw)
