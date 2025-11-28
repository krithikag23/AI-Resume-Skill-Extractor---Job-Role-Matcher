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

# ---------------- ROLE → SKILL DATASET ----------------

# Built-in skill database for each role (you can expand this later)
ROLE_SKILLS = {
    "Data Scientist": [
        "python", "r", "sql", "statistics", "probability", "machine learning",
        "deep learning", "pandas", "numpy", "scikit-learn", "data visualization",
        "matplotlib", "seaborn", "power bi", "tableau", "feature engineering",
        "hypothesis testing", "regression", "classification", "clustering"
    ],
    "Machine Learning Engineer": [
        "python", "java", "c++", "mlops", "docker", "kubernetes", "tensorflow",
        "pytorch", "scikit-learn", "model deployment", "feature engineering",
        "ci/cd", "aws", "gcp", "azure", "api", "rest", "microservices",
        "distributed systems"
    ],
    "Full Stack Developer": [
        "javascript", "typescript", "react", "angular", "vue", "html", "css",
        "node.js", "express", "django", "flask", "spring", "rest api",
        "postgresql", "mysql", "mongodb", "git", "testing", "docker"
    ],
    "Frontend Engineer": [
        "javascript", "typescript", "react", "next.js", "vue", "angular",
        "html", "css", "sass", "responsive design", "ui", "ux",
        "webpack", "vite", "testing", "jest", "cypress"
    ],
    "Backend Engineer": [
        "python", "java", "c#", "node.js", "express", "spring", "django",
        "flask", "rest api", "graphql", "microservices", "sql",
        "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes"
    ],
    "Cloud / DevOps Engineer": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ansible", "ci/cd", "jenkins", "gitlab ci", "linux", "bash",
        "networking", "monitoring", "prometheus", "grafana"
    ],
    "AI Researcher": [
        "python", "pytorch", "tensorflow", "deep learning", "neural networks",
        "transformers", "nlp", "computer vision", "reinforcement learning",
        "research", "paper writing", "experiments", "mathematics", "linear algebra",
        "optimization", "probability", "statistics"
    ],
    "Business Analyst": [
        "excel", "sql", "power bi", "tableau", "data visualization",
        "requirements gathering", "stakeholder management", "storytelling",
        "business analysis", "kpis", "dashboards", "documentation",
        "process improvement"
    ],
    "Cybersecurity Analyst": [
        "network security", "firewalls", "ids", "ips", "siem", "incident response",
        "vulnerability assessment", "penetration testing", "linux",
        "wireshark", "nmap", "threat hunting", "risk assessment", "iso 27001"
    ],
    "Mobile App Developer": [
        "android", "kotlin", "java", "ios", "swift", "flutter", "react native",
        "ui", "ux", "rest api", "firebase", "play store", "app store",
        "mobile testing"
    ],
}