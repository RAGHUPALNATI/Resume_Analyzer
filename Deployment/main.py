import pandas as pd
import numpy as np
from io import BytesIO
import spacy
import docx2txt
import pdfplumber
import pickle as pk
import re
import sklearn
import PyPDF2
import nltk
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

app = FastAPI(title="Resume Analyzer API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
try:
    model = pk.load(open('ModelRFC.pkl', 'rb'))
    Vectorizer = pk.load(open('VECTOR.pkl', 'rb'))
except Exception as e:
    print(f"Error loading models: {e}")

ROLE_KEYWORDS = {
    'Frontend': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 'frontend', 'ui', 'ux', 'web developer'],
    'Backend': ['python', 'java', 'node', 'express', 'django', 'flask', 'spring', 'sql', 'backend', 'api'],
    'Full Stack': ['frontend', 'backend', 'full stack', 'mern', 'mean', 'react', 'node', 'django', 'vue', 'database'],
    'DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform', 'ansible', 'devops'],
    'AI Engineer': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'ai', 'artificial intelligence'],
    'Android': ['android', 'kotlin', 'java', 'mobile', 'app developer', 'android studio'],
    'iOS': ['ios', 'swift', 'objective-c', 'xcode', 'apple', 'mobile'],
    'Data Analyst': ['excel', 'sql', 'tableau', 'powerbi', 'data analysis', 'statistics', 'dashboard'],
    'Data Engineer': ['etl', 'spark', 'hadoop', 'kafka', 'data pipeline', 'airflow', 'redshift', 'bigquery'],
    'Machine Learning': ['machine learning', 'scikit-learn', 'tensorflow', 'pytorch', 'model', 'predictive', 'algorithm'],
    'Blockchain': ['blockchain', 'solidity', 'smart contract', 'ethereum', 'web3', 'crypto'],
    'QA': ['qa', 'testing', 'selenium', 'cypress', 'automation', 'manual testing', 'quality assurance'],
    'Cyber Security': ['security', 'penetration testing', 'firewall', 'ceh', 'cissp', 'vulnerability', 'network security'],
    'UX Design': ['figma', 'sketch', 'adobe xd', 'wireframe', 'prototype', 'user experience', 'user interface'],
    'Game Developer': ['unity', 'unreal', 'c#', 'c++', 'game design', '3d', 'gaming'],
    'Product Manager': ['product management', 'agile', 'scrum', 'roadmap', 'jira', 'stakeholder', 'strategy'],
    'PeopleSoft': ['peoplesoft', 'hcm', 'fscm', 'ps'],
    'SQL Developer': ['sql', 'pl/sql', 'oracle', 'mysql', 'database', 'trigger', 'procedure'],
    'React JS Developer': ['react', 'redux', 'javascript', 'frontend', 'jsx', 'component'],
    'Workday': ['workday', 'hcm', 'integration', 'studio', 'eib', 'report']
}

def predict_role_by_keywords(text, ml_prediction):
    text_lower = text.lower()
    best_role = ml_prediction
    max_score = 0
    
    for role, keywords in ROLE_KEYWORDS.items():
        score = sum([1 for kw in keywords if kw in text_lower])
        if score > max_score and score >= 2:
            max_score = score
            best_role = role
            
    return best_role

def extract_skills(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    tokens = [token.text for token in nlp_text if not token.is_stop]
    
    skills_path = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'skills.csv')
    try:
        data = pd.read_csv(skills_path)
        skills = list(data.columns.values)
    except:
        skills = [] # Fallback if not found
        
    skillset = []
    
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)
            
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def getText(file_content, filename):
    fullText = ''
    if filename.endswith('.docx'):
        doc = docx2txt.process(BytesIO(file_content))
        fullText = doc
    elif filename.endswith('.pdf'):
        pdoc = PyPDF2.PdfReader(BytesIO(file_content))
        for page in pdoc.pages:
            fullText += page.extract_text()
    return fullText

def display(file_content, filename):
    resume = []
    if filename.endswith('.docx'):
        resume.append(docx2txt.process(BytesIO(file_content)))
    elif filename.endswith('.pdf'):
        # Using pdfplumber for display extract
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            pages = pdf.pages[0]
            resume.append(pages.extract_text())
    return resume

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
    
    file_content = await file.read()
    
    try:
        # Get raw text
        extText = getText(file_content, file.filename)
        
        # Get text for preprocessing
        dispText = display(file_content, file.filename)
        cleaned = preprocess(dispText)
        
        # Predict using ML model
        ml_prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        
        # Keyword refinement
        final_prediction = predict_role_by_keywords(extText, ml_prediction)
        
        # Extract skills
        extracted_skills = extract_skills(extText)
        
        return {
            "filename": file.filename,
            "predicted_profile": final_prediction,
            "skills": extracted_skills
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
