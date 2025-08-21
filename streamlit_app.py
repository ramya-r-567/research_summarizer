import streamlit as st
import io
import PyPDF2
import textwrap
from transformers import pipeline

st.set_page_config("Research Summarizer", page_icon="📜", layout="wide")

st.title("Research Paper Summarizer & Classifier")
st.caption("Paste Text or Upload a PDF. Summarize and Classify with AI.")

def extract_text_from_pdf(file) -> str:
    """Extract text from upload PDF"""
    pdf_reader= PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, max_words=250):
    """Split text from upload"""
    words = text.split()
    return[" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ---------------------- Load Models ----------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

@st.cache_resource
def load_classifer():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

summarizer = load_summarizer()
classifier = load_classifer()

col1, col2 = st.columns(2)

with col1:
    text_input = st.text_area("Paste research text here:", height=100)
    
with col2:
    upload_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    pdf_text = ""
    if upload_file is not None:
        pdf_text = extract_text_from_pdf(io.BytesIO(upload_file.read()))
        st.success("PDF uploaded and text extracted ✅")

final_text = text_input if text_input else pdf_text

# ---------------------- Buttons ----------------------
summarize_btn = st.button("Summarize")
classify_btn = st.button("Classify")
clear_btn = st.button("Clear")


# ---------------------- Output----------------------
#define summary length settings
max_len = 250
min_len = 50

summary = ""
classification_result = None

if summarize_btn and final_text:
    st.write("🔍 Generating summary...")
    chunks = chunk_text(final_text, max_words=250)
    st.subheader("📝 Summary")
    summary_parts = []
    progress = st.progress(0)
    for i , chunk in enumerate(chunks,1):
        part = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        summary_parts.append(part)
        progress.progress(i/len(chunks))
    summary = " ".join(summary_parts)
    st.write(summary)

#define categories/labels
labels = ["Biotechnology", "Artifical Intelligence", "Biology", "Physics", "Computr Science", "Medicine", "Mathematics", "Engineering"]

if classify_btn and final_text:
    st.write("📊 Classifying text...")
    st.subheader("🏷️ Classification")
    classification_result = classifier(final_text[:800], candidate_labels=labels)
    st.success(f"Top category: {classification_result['labels'][0]} ({classification_result['score'][0]:.2f})")
    st.write("### Confidence scores")
    for lbl, score in zip(classification_result['label'], classification_result["scores"]):
        st.write(f"- {lbl}: {score:2f}")

