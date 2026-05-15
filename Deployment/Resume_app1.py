# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import click
import spacy
import docx2txt
import pdfplumber
from pickle import load
import requests
import re
import os
import sklearn
import PyPDF2
import nltk
import pickle as pk
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
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot  as plt
stop=set(stopwords.words('english'))
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
from sklearn.feature_extraction.text import TfidfVectorizer

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
        # Only override ML prediction if we find strong evidence (at least 2 matching keywords)
        if score > max_score and score >= 2:
            max_score = score
            best_role = role
            
    return best_role

#----------------------------------------------------------------------------------------------------

st.title('             RESUME CLASSIFICATION     ')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)

st.subheader('Hey, Welcome')

# FUNCTIONS
def extract_skills(resume_text):

    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks

    tokens = [token.text for token in nlp_text if not token.is_stop] # removing stop words and implementing word tokenization
            
    
    data = pd.read_csv(r"..\Datasets\skills.csv") # reading the csv file
            
    
    skills = list(data.columns.values)# extract values
            
    skillset = []
            
    
    for token in tokens:                 # check for one-grams (example: python)
        if token.lower() in skills:
            skillset.append(token)
            
   
    for token in noun_chunks:            # check for bi-grams and tri-grams (example: machine learning)
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def getText(filename):
      
    # Create empty string 
    fullText = ''
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        
        for para in doc:
            fullText = fullText + para
            
           
    else:  
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfReader(filename)
            number_of_pages = len(pdoc.pages)
            page = pdoc.pages[0]
            page_content = page.extract_text()
             
        for paragraph in page_content:
            fullText =  fullText + paragraph
         
    return (fullText)


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))

    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
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

file_type=pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile','Skills',])
filename = []
predicted = []
skills = []

#-------------------------------------------------------------------------------------------------
# MAIN CODE
import pickle as pk
model = pk.load(open(r'ModelRFC.pkl', 'rb'))
Vectorizer = pk.load(open(r'VECTOR.pkl', 'rb'))

upload_file = st.file_uploader('Upload Your Resumes',
                                type= ['docx','pdf'],accept_multiple_files=True)
  
for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned=preprocess(display(doc_file))
        extText = getText(doc_file)
        ml_prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        prediction = predict_role_by_keywords(extText, ml_prediction)
        predicted.append(prediction)
        skills.append(extract_skills(extText))
        
if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())
    
    select = list(ROLE_KEYWORDS.keys()) + ['Internship', 'Peoplesoft Admin', 'Peoplesoft DBA', 'Peoplesoft Finance', 'Peoplesoft FSCM']
    select = sorted(list(set(select)))
    st.subheader('Select as per Requirement')
    option = st.selectbox('Fields', select)

    st.table(file_type[file_type['Predicted Profile'] == option])