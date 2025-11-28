import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# PDF parsing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Skill Extractor & Role Matcher",
    layout="wide"
)

st.title("AI Resume Skill Extractor & Job Role Matcher")
st.write(
    "Upload your resume and this app will extract skills and match you to the most relevant job roles "
    "using an NLP-based similarity model."
)