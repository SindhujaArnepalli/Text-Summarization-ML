import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# --- NLTK setup (handle punkt_tab and fallback to punkt) ---
import nltk

def ensure_nltk_tokenizers():
    # Prefer modern resource on NLTK >= 3.8.2, but accept legacy if present
    for res in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(res)
            return
        except LookupError:
            pass

    # Try to download both quietly; ignore failures and recheck
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

    for res in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(res)
            return
        except LookupError:
            continue

    raise RuntimeError("NLTK tokenizers not available (punkt_tab/punkt).")

ensure_nltk_tokenizers()

# --- Sumy imports (after ensuring NLTK data) ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# --- Model load ---
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # real tensors in CPU envs
    device_map=None              # avoid accidental meta/device mapping
)

st.title("Customer Review Summarizer")

# User input
review = st.text_area("Enter customer review here:")

@st.cache_data(show_spinner=False)
def bart_summary(text: str) -> str:
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=150,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@st.cache_data(show_spinner=False)
def textrank_summary(text: str, sentences_count: int = 2) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(s) for s in summary)

if st.button("Summarize"):
    if review.strip():
        bart_result = bart_summary(review)
        textrank_result = textrank_summary(review)

        st.subheader("🔹 BART Summary (Abstractive)")
        st.write(bart_result)

        st.subheader("🔹 TextRank Summary (Extractive)")
        st.write(textrank_result)

# Keep model on CPU by default (change to 'cuda' if GPU is available)
device = torch.device("cpu")
model.to(device)
