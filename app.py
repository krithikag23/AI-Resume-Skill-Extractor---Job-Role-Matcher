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

# Flatten global skill vocabulary for detection
GLOBAL_SKILLS = sorted({s.lower() for skills in ROLE_SKILLS.values() for s in skills})

# ---------------- HELPERS ----------------

def read_pdf(file) -> str:
    if PyPDF2 is None:
        st.warning("PyPDF2 is not installed. Please install it or upload a .txt file.")
        return ""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    
def normalize_text(text: str) -> str:
    return text.lower()

def extract_skills_from_text(text: str, skill_vocab=None):
    if skill_vocab is None:
        skill_vocab = GLOBAL_SKILLS
    text_low = text.lower()
    found = set()
    for skill in skill_vocab:
        if skill in text_low:
            found.add(skill)
    return sorted(found)

def build_role_corpus(role_skills_dict):
    """
    Build a small text corpus: one document per role, consisting of its skills.
    """
    roles = []
    docs = []
    for role, skills in role_skills_dict.items():
        roles.append(role)
        docs.append(" ".join(skills))
    return roles, docs

def compute_role_similarity(resume_skills, roles, role_docs):
    """
    Fit a TF-IDF model on [resume_doc] + role_docs and compute cosine similarity.
    """
    resume_doc = " ".join(resume_skills) if resume_skills else ""
    corpus = [resume_doc] + role_docs  # index 0 = resume
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    resume_vec = tfidf_matrix[0:1]
    role_vecs = tfidf_matrix[1:]

    sims = cosine_similarity(resume_vec, role_vecs).flatten()
    return sims


# ---------------- SIDEBAR ----------------

st.sidebar.header("Upload & Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload your resume (.pdf or .txt)",
    type=["pdf", "txt"]
)


use_sample = st.sidebar.checkbox("Use sample resume text", value=not bool(uploaded_file))

min_skill_threshold = st.sidebar.slider(
    "Minimum number of skills to consider a match",
    min_value=0,
    max_value=20,
    value=5,
    step=1
)


# ---------------- MAIN FLOW ----------------

if uploaded_file is not None and not uploaded_file.name.endswith(".txt"):
    st.sidebar.info(f"Uploaded file: {uploaded_file.name}")

if uploaded_file or use_sample:
    # 1. Read text
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            resume_text = read_pdf(uploaded_file)
    else:
        st.info("Using sample resume text (you can upload your own PDF or TXT in the sidebar).")
        resume_text = """
        Experienced Machine Learning Engineer with 2+ years of experience in building and deploying ML models.
        Skilled in Python, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, Docker, Kubernetes and REST APIs.
        Worked on end-to-end ML pipelines on AWS and GCP, including data preprocessing, feature engineering,
        model training, evaluation and deployment using CI/CD. Familiar with deep learning, NLP and computer vision.
        """

    # 2. Show raw text (collapsible)
    with st.expander("📄 View Parsed Resume Text"):
        st.write(resume_text[:5000])  # limit just in case

    # 3. Extract skills
    extracted_skills = extract_skills_from_text(resume_text, GLOBAL_SKILLS)

    st.subheader("🔍 Extracted Skills from Resume")
    if extracted_skills:
        st.write(f"Total skills detected: **{len(extracted_skills)}**")
        st.write(", ".join(sorted(set(extracted_skills))))
    else:
        st.warning("No known skills were detected. Try adjusting your resume or expanding the skill vocabulary.")
    
    # 4. Build role corpus & compute similarity
    roles, role_docs = build_role_corpus(ROLE_SKILLS)
    similarities = compute_role_similarity(extracted_skills, roles, role_docs)

    # 5. Build result dataframe
    results_df = pd.DataFrame({
        "Role": roles,
        "Similarity": similarities
    }).sort_values("Similarity", ascending=False)

    # For each role, also compute overlap count
    overlap_counts = []
    for role in results_df["Role"]:
        role_skill_set = set(ROLE_SKILLS[role])
        overlap_counts.append(len(role_skill_set.intersection(extracted_skills)))
    results_df["Matched Skills"] = overlap_counts

    # Filter by minimum skill threshold if set
    filtered_df = results_df[results_df["Matched Skills"] >= min_skill_threshold]

    st.subheader("🎯 Role Match Ranking")

    if filtered_df.empty:
        st.warning("No roles meet the minimum matched skill threshold. Try lowering it in the sidebar.")
    else:
        # Highlight top role
        top_row = filtered_df.iloc[0]
        st.markdown(
            f"**Best Matched Role:** `{top_row['Role']}`  \n"
            f"Similarity score: **{top_row['Similarity']:.3f}**  |  "
            f"Matched skills: **{top_row['Matched Skills']}**"
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Role Match Scores (for screenshots)**")
            fig = px.bar(
                filtered_df,
                x="Role",
                y="Similarity",
                title="Similarity of Resume to Each Role (TF-IDF + Cosine)",
                labels={"Similarity": "Similarity Score", "Role": "Job Role"}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Table View**")
            st.dataframe(
                filtered_df.reset_index(drop=True).style.format(
                    {"Similarity": "{:.3f}"}
                )
            )

        # 6. Detailed view per role
        st.subheader("🧩 Role-wise Skill Breakdown")

        selected_role = st.selectbox(
            "Select a role to inspect",
            options=results_df["Role"].tolist(),
            index=0
        )

        role_skill_set = set(ROLE_SKILLS[selected_role])
        matched = sorted(role_skill_set.intersection(extracted_skills))
        missing = sorted(role_skill_set.difference(extracted_skills))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"✅ **Matched skills for {selected_role}:**")
            if matched:
                st.write(", ".join(matched))
            else:
                st.write("None detected.")

        with c2:
            st.markdown(f"📌 **Recommended to add / highlight for {selected_role}:**")
            if missing:
                st.write(", ".join(missing))
            else:
                st.write("You already list most skills for this role!")
else:
    st.info("Upload a resume (.pdf or .txt) or enable **'Use sample resume text'** from the sidebar to begin.")                