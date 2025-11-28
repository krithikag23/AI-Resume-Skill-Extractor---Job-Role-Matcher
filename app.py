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