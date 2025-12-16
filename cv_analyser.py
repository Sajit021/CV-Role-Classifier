

# Installing required dependencies

!pip install pdfplumber pymupdf pytesseract spacy nltk scikit-learn joblib sentence-transformers
!python -m spacy download en_core_web_sm

# IMPORT LIBRARIES

import os
import numpy as np
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# DOWNLOAD AND LOAD NLTK STOPWORDS

nltk.download('stopwords')  # download once per session
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# LOAD SPACY NLP MODEL
nlp = spacy.load("en_core_web_sm")

## just for checking
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("John Doe is a Data Scientist with 5 years of experience.")
for ent in doc.ents:
    print(ent.text, ent.label_)

from google.colab import files

# Upload the ZIP file
uploaded = files.upload()

# Unzip into /content/cvs
!unzip /content/CV_Dataset.zip -d /content/cvs

# Check contents
os.listdir("/content/cvs")

# PDF text extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# Text cleaning + preprocessing
def clean_cv_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)            # remove emails
    text = re.sub(r'\+?\d[\d\s\-]{8,}', ' ', text) # remove phone numbers
    text = re.sub(r'[^a-z\s]', ' ', text)          # remove non-alphabetic chars
    text = re.sub(r'\s+', ' ', text).strip()       # remove extra spaces

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc
              if token.is_alpha and token.text not in stop_words and len(token.text) > 2]
    return " ".join(tokens)

CV_FOLDER = "/content/cvs/CV_Dataset" # using the dataset under new name
cv_data = []

for filename in os.listdir(CV_FOLDER):
    if filename.lower().endswith(".pdf"):  # handles .PDF too
        file_path = os.path.join(CV_FOLDER, filename)
        raw_text = extract_text_from_pdf(file_path)
        cleaned_text = clean_cv_text(raw_text)
        cv_data.append({
            "file_name": filename,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text
        })

df = pd.DataFrame(cv_data)
print(df.shape)
df.head()

# Define roles and their required skills

roles_skills = [
    {"skills": ["Python", "TensorFlow", "Machine Learning"], "label": "Machine Learning Engineer"},
    {"skills": ["NLP", "TF-IDF", "Text Classification"], "label": "NLP Engineer"},
    {"skills": ["SQL", "pandas", "Power BI"], "label": "Data Analyst"},
    {"skills": ["Java", "SQL", "REST APIs"], "label": "Backend Developer"},
    {"skills": ["HTML", "CSS", "JavaScript"], "label": "Web Developer"},
    {"skills": ["Linux", "Networking", "Cloud"], "label": "Systems Engineer"},
    {"skills": ["Software Testing", "Selenium"], "label": "QA Engineer"},
    {"skills": ["MERN Stack", "APIs", "Git"], "label": "Full-Stack Developer"},
    {"skills": ["Troubleshooting", "User Support"], "label": "IT Support Analyst"},
    {"skills": ["Excel", "SQL", "Process Analysis"], "label": "Business Analyst"}
]

def assign_role_from_skills(text):
    """
    Assign roles if CV has 2+ matching skills per role.
    Returns comma-separated roles or 'Other'.
    """
    text_lower = text.lower()
    eligible_roles = []

    for role_dict in roles_skills:
        skills = role_dict["skills"]
        label = role_dict["label"]
        match_count = sum(1 for skill in skills if skill.lower() in text_lower)
        if match_count >= 2:
            eligible_roles.append(label)

    return ", ".join(eligible_roles) if eligible_roles else "Other"

# Apply to DataFrame
df['role'] = df['cleaned_text'].apply(assign_role_from_skills)
df[['file_name', 'role']].head(10)

from sklearn.naive_bayes import MultinomialNB  #importing to train model for small dataset

# Keep only CVs with a single role
df_train = df[df['role'] != "Other"]
df_train = df_train[df_train['role'].str.contains(",") == False]

X = df_train['cleaned_text']
y = df_train['role']

# Train-test split (no stratify for small dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization (reduce features for small dataset)
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

!ls /content/cvs # I wanted to check if my dataset is working or not

!ls /content/cvs/CV_Dataset

# Example for predicting a new CV
new_cv_file = "/content/cvs/CV_Dataset/SahajidRahaman_CV.pdf"  # replace with your file
raw_text = extract_text_from_pdf(new_cv_file)
clean_text = clean_cv_text(raw_text)

# 1. Skill-based role assignment
predicted_roles_skills = assign_role_from_skills(clean_text)
print("Skill-based predicted role(s):", predicted_roles_skills)

# 2. ML-based prediction
vec = vectorizer.transform([clean_text])
predicted_role_ml = clf.predict(vec)[0]
print("ML-based predicted role:", predicted_role_ml)

#saving the model

joblib.dump(clf, "cv_role_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")