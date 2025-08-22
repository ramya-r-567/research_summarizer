import streamlit as st
import io
import textwrap
from transformers import pipeline
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# load your custom css
local_css("style.css")

# ✅ Safe PDF import (works on both VS Code and Streamlit Cloud)
try:
    from pypdf import PdfReader   # preferred modern library
except ImportError:
    from PyPDF2 import PdfReader  # fallback if pypdf fails

st.set_page_config("Research Summarizer", page_icon="📜", layout="wide")

st.title("Research Paper Summarizer & Classifier")
st.caption("Paste Text or Upload a PDF. Summarize and Classify with AI.")

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, max_words=300):
    """Split text from upload"""
    words = text.split()
    return[" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ---------------------- Load Models ----------------------
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # small & fast

def load_classifier():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")  # lighter than bart-large


with st.spinner("🚀 Loading AI models... (first time may take 1–2 minutes)"):
    summarizer = load_summarizer()
    classifier = load_classifier()


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
if len(final_text.split()) > 3000:
    st.warning("⚠️ Input is very long! Summarization may be slow. Try pasting only the abstract or introduction.")

# ---------------------- Buttons ----------------------
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1,1,1,1])

with col_btn1:
    summarize_btn = st.button("📝 Summarize")

with col_btn2:
    classify_btn = st.button("📊 Classify")

with col_btn3:
    clear_btn = st.button("🧹 Clear")


# ---------------------- Output----------------------
#define summary length settings
max_len = 150
min_len = 30

summary = ""
classification_result = None

final_text = user_input.strip() if user_input else ""

if summarize_btn and final_text:
    st.write("🔍 Generating summary...")
    chunks = chunk_text(final_text, max_words=300)
    st.subheader("📝 Summary")
    summary_parts = []
    progress = st.progress(0)
    for i , chunk in enumerate(chunks,1):
        part = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        summary_parts.append(part)
        progress.progress(i/len(chunks))
    summary = " ".join(summary_parts)
    

      # ✅ Keep summary in session_state so it doesn't disappear
    st.session_state.summary = summary

#define categories/labels
labels = ["Biotechnology", "Artifical Intelligence", "Biology", "Physics", "Computr Science", "Medicine", "Mathematics", "Engineering"]

if classify_btn and final_text:
    st.write("📊 Classifying text...")
    st.subheader("🏷️ Classification")
    classification_result = classifier(final_text[:800], candidate_labels=labels)
    st.success(f"Top category: {classification_result['labels'][0]} ({classification_result['scores'][0]:.2f})")
    st.write("### Confidence scores")
    for lbl, score in zip(classification_result['labels'], classification_result["scores"]):
        st.write(f"- {lbl}: {score:2f}")

#-----------------Download Results-----------------

# Function to generate PDF
def generate_pdf(summary_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Research Summary", styles['Heading1']))
    story.append(Spacer(1, 12))

    for line in summary_text.split("\n"):
        story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------- MAIN APP ----------------

# Example: after your model generates summary, save it in session_state
if "summary" not in st.session_state:
    st.session_state.summary = ""   # initialize

# Let's assume you already have a variable summary
# After summarization, set it like this:
# st.session_state.summary = summary

# Show the summary if it exists
if st.session_state.summary:
    st.write(st.session_state.summary)

    pdf_buffer = generate_pdf(st.session_state.summary)

    st.download_button(
        label="📄 Download Summary as PDF",
        data=pdf_buffer,
        file_name="summary.pdf",
        mime="application/pdf"
    )
