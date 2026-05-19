import pandas as pd
import numpy as np
import re
import pickle as pk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import os

# Download NLTK data if missing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Text cleaning function
def preprocess(sentence):
    sentence = str(sentence).lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if len(w) > 2 and w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'Resume.csv')
    
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the Resume.csv file is placed in the Datasets folder.")
        return

    df = pd.read_csv(dataset_path)
    
    if 'Resume' not in df.columns:
        # Some datasets use 'Resume_str' or 'text'
        for col in ['Resume_str', 'text', 'Text']:
            if col in df.columns:
                df.rename(columns={col: 'Resume'}, inplace=True)
                break

    print("Cleaning resume text...")
    df['Cleaned_Resume'] = df['Resume'].apply(preprocess)
    
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    # This generates 384-dimensional embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = embedder.encode(df['Cleaned_Resume'].tolist(), show_progress_bar=True)
    
    print("Training KMeans clustering model (12 clusters)...")
    num_clusters = 12
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    df['Cluster'] = kmeans.labels_
    
    print("\nCluster Distribution Summary:")
    print(df['Cluster'].value_counts().sort_index())
    
    print("\nSaving models to disk...")
    # We save the KMeans model. 
    # For the embedder, we can either save the model name or the entire object.
    # Saving the name is safer, but user asked to save embedder_model.pkl
    with open('cluster_model.pkl', 'wb') as f:
        pk.dump(kmeans, f)
        
    with open('embedder_model.pkl', 'wb') as f:
        pk.dump(embedder, f)
        
    print("Models saved successfully: cluster_model.pkl, embedder_model.pkl")

if __name__ == "__main__":
    main()
