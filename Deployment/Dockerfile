FROM python:3.10-slim

WORKDIR /app

COPY Requirements.txt .

RUN pip install --no-cache-dir -r Requirements.txt

# Download NLTK data during build to avoid downloading on startup
RUN python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words wordnet stopwords omw-1.4

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
