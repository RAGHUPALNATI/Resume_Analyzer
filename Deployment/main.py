import pandas as pd
import numpy as np
from io import BytesIO
import docx2txt
import pdfplumber
import pickle as pk
import re
import sklearn
import PyPDF2
import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from skills_extractor import extract_skills_from_text

app = FastAPI(title="Resume Analyzer Dual-ML API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
print("Loading models...")
model_rfc = None
vectorizer = None
cluster_model = None
embedder_model = None

try:
    with open('ModelRFC.pkl', 'rb') as f:
        model_rfc = pk.load(f)
    with open('VECTOR.pkl', 'rb') as f:
        vectorizer = pk.load(f)
    print("Supervised ML models loaded successfully.")
except Exception as e:
    print(f"Warning: Supervised ML models not found: {e}")

try:
    with open('cluster_model.pkl', 'rb') as f:
        cluster_model = pk.load(f)
    with open('embedder_model.pkl', 'rb') as f:
        embedder_model = pk.load(f)
    print("Unsupervised ML models loaded successfully.")
except Exception as e:
    print(f"Warning: Unsupervised ML models not found. Run clustering.py first: {e}")

# Text cleaning function
def preprocess(sentence):
    sentence = str(sentence).lower()
    sentence = sentence.replace('{html}', "") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    
    # Simple regex tokenizer to avoid full NLTK dependency if possible, 
    # but maintaining the same preprocessing logic
    tokens = re.findall(r'\w+', rem_num)
    
    # Very basic stop word list to keep it fast and standalone if NLTK fails
    basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    filtered_words = [w for w in tokens if len(w) > 2 and w not in basic_stopwords]
    
    return " ".join(filtered_words)

def getText(file_content, filename):
    fullText = ''
    try:
        if filename.endswith('.docx'):
            doc = docx2txt.process(BytesIO(file_content))
            fullText = doc
        elif filename.endswith('.pdf'):
            # Using pdfplumber as it is robust
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        fullText += text + "\n"
    except Exception as e:
        print(f"Error extracting text: {e}")
    return fullText

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_old(file: UploadFile = File(...)):
    """Legacy endpoint for backward compatibility"""
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
    
    file_content = await file.read()
    
    try:
        extText = getText(file_content, file.filename)
        cleaned = preprocess(extText)
        
        ml_prediction = "Unknown"
        if model_rfc and vectorizer:
            ml_prediction = model_rfc.predict(vectorizer.transform([cleaned]))[0]
            
        return {
            "filename": file.filename,
            "predicted_profile": ml_prediction,
            "message": "Please use /analyze for the full Dual-ML analysis."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(None), text: str = Form(None)):
    """Dual-ML analysis endpoint. Accepts a file or raw text."""
    
    extText = ""
    
    # 1. Extract text from file or form data
    if file:
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
        file_content = await file.read()
        extText = getText(file_content, file.filename)
    elif text:
        extText = text
    else:
        raise HTTPException(status_code=400, detail="Please provide a file or raw text.")
        
    if not extText.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the input.")

    cleaned_text = preprocess(extText)
    
    # Outputs initialization
    predicted_category = "Analysis Pending"
    cluster_group = "Unclustered"
    cluster_confidence = 0
    
    # 2. Supervised ML: Predict Category using Random Forest
    if model_rfc and vectorizer:
        try:
            predicted_category = model_rfc.predict(vectorizer.transform([cleaned_text]))[0]
        except Exception as e:
            print(f"Prediction error: {e}")
    else:
        predicted_category = "RF Model Not Loaded"

    # 3. Unsupervised ML: Predict Cluster using KMeans & SentenceTransformers
    if cluster_model and embedder_model:
        try:
            embedding = embedder_model.encode([cleaned_text])
            cluster_id = cluster_model.predict(embedding)[0]
            cluster_group = f"Cluster {cluster_id} — Profile Type {cluster_id}"
            
            # Calculate Confidence based on distance to cluster center
            center = cluster_model.cluster_centers_[cluster_id]
            distance = np.linalg.norm(embedding[0] - center)
            # Normalize distance to a confidence percentage (heuristic)
            # Distance typically ranges from 0 to 2 for normalized embeddings
            confidence = max(0, min(100, int(100 - (distance * 40))))
            cluster_confidence = confidence
        except Exception as e:
            print(f"Clustering error: {e}")
            cluster_group = "Clustering Failed"
    else:
        cluster_group = "Cluster Model Not Loaded"

    # 4. Extract Skills using NLP
    skills_data = extract_skills_from_text(extText, predicted_category)

    # 5. Compile the final JSON response
    return {
        "predicted_category": predicted_category,
        "cluster_group": cluster_group,
        "cluster_confidence": cluster_confidence,
        "top_skills": skills_data["top_skills"],
        "match_score": skills_data["match_score"],
        "missing_skills": skills_data["missing_skills"]
    }
