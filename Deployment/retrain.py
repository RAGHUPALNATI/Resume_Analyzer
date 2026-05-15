import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('../Datasets/Cleaned_Resumes.csv')
df['Clean_Text'] = df['Clean_Text'].fillna('')
X = df['Clean_Text'].values
y = df['Category'].values

# Vectorizer
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_vec, y)

# Save
with open('VECTOR.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('ModelRFC.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Retraining completed.")
