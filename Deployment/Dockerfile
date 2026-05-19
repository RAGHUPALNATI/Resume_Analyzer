FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for some ML packages (e.g., umap-learn, scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY Requirements.txt .

RUN pip install --no-cache-dir -r Requirements.txt

# Download NLTK data during build
RUN python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words wordnet stopwords omw-1.4

# Pre-download spaCy model
RUN python -m spacy download en_core_web_sm

# Set environment variable to avoid huggingface/transformers downloading at runtime
ENV TRANSFORMERS_CACHE=/app/.cache
# Download the SentenceTransformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
