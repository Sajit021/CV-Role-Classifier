# CV Role Classification using NLP & ML

## Project Overview

This project is designed to **automatically classify the role a candidate has applied for based on their CV**. It is particularly useful for recruiters to quickly shortlist candidates according to skills listed in their CVs.

The system supports:

- Extraction of text from PDF CVs
- Cleaning and preprocessing of CV text
- Skill-based role assignment for multiple eligible roles
- Optional machine learning (ML) model for role prediction on new CVs

---

## Technologies & Libraries

- **Python 3.12**  
- **PDF Processing:** `pdfplumber`, `PyMuPDF` (`fitz`)  
- **NLP & Preprocessing:** `spaCy`, `NLTK`  
- **Machine Learning:** `scikit-learn` (TF-IDF vectorizer, Naive Bayes classifier)  
- **Data Handling:** `pandas`, `numpy`  
- **Visualization:** `matplotlib`, `seaborn`  
- **Model Persistence:** `joblib`  

---

## Key Features

### PDF CV Upload & Extraction
- Supports multiple CVs stored in a folder or ZIP file  
- Extracts text from PDFs using `pdfplumber`  

### Text Cleaning & Preprocessing
- Converts text to lowercase  
- Removes emails, phone numbers, and non-alphabetic characters  
- Performs stopword removal and lemmatization using `spaCy` and `NLTK`  

### Skill-Based Role Assignment
- Assigns roles based on a predefined skills-to-role mapping  
- Requires **2+ matching skills** to assign a role  
- Supports multiple role assignments per CV  
- Assigns `"Other"` if no role matches  

### Optional ML-Based Prediction
- TF-IDF vectorization of cleaned CV text  
- Multinomial Naive Bayes classifier  
- Can predict roles for new CVs not in the training set  
- Includes evaluation metrics: accuracy, classification report, confusion matrix  

### New CV Prediction
- Can predict roles for any new CV PDF  
- Returns both **skill-based prediction** and **ML-based prediction**  

---

## Skills-to-Role Mapping

| Role | Required Skills (2+ required) |
|------|-------------------------------|
| Machine Learning Engineer | Python, TensorFlow, Machine Learning |
| NLP Engineer | NLP, TF-IDF, Text Classification |
| Data Analyst | SQL, pandas, Power BI |
| Backend Developer | Java, SQL, REST APIs |
| Web Developer | HTML, CSS, JavaScript |
| Systems Engineer | Linux, Networking, Cloud |
| QA Engineer | Software Testing, Selenium |
| Full-Stack Developer | MERN Stack, APIs, Git |
| IT Support Analyst | Troubleshooting, User Support |
| Business Analyst | Excel, SQL, Process Analysis |

---

## Project Workflow

1. **Upload CVs:** Upload a folder or ZIP of PDF CVs to Colab.  
2. **Text Extraction:** Extract text from PDFs using `pdfplumber`.  
3. **Text Cleaning:** Lowercase, remove unwanted characters, lemmatize, remove stopwords.  
4. **Role Assignment:** Assign roles using skill-based mapping.  
5. **Optional ML Training:** Train TF-IDF + Naive Bayes classifier for additional predictions.  
6. **Predict New CVs:** Extract, clean, and predict roles for new CVs.  
7. **Model Saving:** Save trained vectorizer and classifier using `joblib`.  

---

## Usage Example

```python
# Predict roles for a new CV
new_cv_file = "/content/cvs/SahajidRahaman_CV.pdf"
raw_text = extract_text_from_pdf(new_cv_file)
clean_text = clean_cv_text(raw_text)

# Skill-based prediction
predicted_roles_skills = assign_role_from_skills(clean_text)
print("Skill-based predicted role(s):", predicted_roles_skills)

# ML-based prediction
vec = vectorizer.transform([clean_text])
predicted_role_ml = clf.predict(vec)[0]
print("ML-based predicted role:", predicted_role_ml)

## Model Persistence

# Save model and vectorizer
joblib.dump(clf, "cv_role_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
