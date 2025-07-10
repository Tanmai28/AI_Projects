import re
import string
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('/emails.csv')
emails = df['text'].tolist()
labels = [1 if label == 'spam' else 0 for label in df['label']]

# Feature Engineering: Detect links and phone numbers
def contains_link(text):
    return int(bool(re.search(r'http[s]?://', text)))

def contains_phone_number(text):
    return int(bool(re.search(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', text)))

# Preprocessing: Lowercase, remove stopwords and punctuation
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http[s]?://\S+", "", text)  # Remove URLs
    text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '', text)  # Remove phone numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Tokenization and Vectorization
processed_emails = [preprocess(email) for email in emails]
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(processed_emails)

# Add engineered features
X_link = np.array([contains_link(email) for email in emails]).reshape(-1, 1)
X_phone = np.array([contains_phone_number(email) for email in emails]).reshape(-1, 1)
X = hstack([X_text, X_link, X_phone])

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict on new email
def predict_spam(email):
    processed = preprocess(email)
    X_new_text = vectorizer.transform([processed])
    X_new_link = np.array([contains_link(email)]).reshape(-1, 1)
    X_new_phone = np.array([contains_phone_number(email)]).reshape(-1, 1)
    X_new = hstack([X_new_text, X_new_link, X_new_phone])
    pred = model.predict(X_new)[0]
    return "SPAM" if pred == 1 else "NOT SPAM"

# Example usage
test_email = "Congratulations! Call 555-123-4567 to claim your free prize now."
print("\nTest email prediction:", predict_spam(test_email))
