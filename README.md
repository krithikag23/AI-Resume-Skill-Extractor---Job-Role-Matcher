# AI Resume Skill Extractor & Job Role Matcher

An interactive **Streamlit** app that:
- Extracts skills from a resume (PDF/TXT)
- Matches the candidate to **10 predefined job roles** using an NLP-based similarity model
- Shows ranked role match scores and a detailed skill gap analysis


## Supported Roles
1. Data Scientist 
2. Machine Learning Engineer  
3. Full Stack Developer 
4. Frontend Engineer 
5. Backend Engineer
6. Cloud / DevOps Engineer
7. AI Researcher
8. Business Analyst  
9. Cybersecurity Analyst
10. Mobile App Developer  


## How It Works
1. **Skill Extraction**
   - The app scans the resume text and detects known skills from a built-in skill vocabulary.
        
2. **Role Profiles**
      - Each job role has a curated list of core skills (our built-in "dataset").

3. **NLP Similarity Model**
      - We create a small corpus with:
          - One document for the resume (extracted skills)
          - One document per role (its skill list)
      - A **TF-IDF Vectorizer** is trained on this corpus.
      - **Cosine similarity** is computed between the resume vector and each role vector.

4. **Dashboard**
      - Bar chart of similarity scores for all roles 
