from flask import Flask, request, render_template, Markup
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import nltk
from lime.lime_text import LimeTextExplainer
nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__)

# Load the necessary components
model = joblib.load('random_forest_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
pca = joblib.load('pca.joblib')

def preprocess(text):
    # Remove tags
    tags_to_remove = ['\n', '\'']
    for tag in tags_to_remove:
        if tag == '\n':
            text = text.replace(tag, ' ')
        else:
            text = text.replace(tag, '')

    # Count punctuation marks
    punctuation_count = sum(1 for char in text if char in string.punctuation)

    # Count linking words
    stop_words = set(stopwords.words('english'))
    additional_linking_words = {'to', 'the', 'and', 'of', 'in', 'on', 'for', 'with', 'at', 'a', 'an'}
    linking_words = stop_words.union(additional_linking_words)
    linking_words_count = sum(1 for word in word_tokenize(text.lower()) if word in linking_words)

    # Extract features
    char_count = len(text)
    word_count = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
    uppercase_count = sum(1 for char in text if char.isupper())
    digit_count = sum(1 for char in text if char.isdigit())

    # Detect slangs and emails, convert detection to 1 if found, else 0
    slang_regex = r'\b(?:lol|brb|omg)\b'
    slangs_detected = 1 if re.findall(slang_regex, text, flags=re.IGNORECASE) else 0

    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_detected = 1 if re.findall(email_regex, text, re.IGNORECASE) else 0

    # New feature: Number of unique words
    num_unique_words = len(set(word_tokenize(text.lower())))

    # Combine all features into a single structure
    features = {
        'char_count': char_count,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'uppercase_count': uppercase_count,
        'digit_count': digit_count,
        'punctuation_count': punctuation_count,
        'linking_words_count': linking_words_count,
        'slangs_detected': slangs_detected,
        'emails_detected': emails_detected,
        'num_unique_words': num_unique_words
    }

    return features

def preprocess_and_extract_features(text):
    preprocessed_features = preprocess(text)
    tfidf_features = vectorizer.transform([text]).toarray()

    # Extracting the values in the correct order
    extracted_features = [
        preprocessed_features[key] for key in [
            'punctuation_count', 'linking_words_count', 'word_count', 'char_count',
            'avg_word_length', 'uppercase_count', 'digit_count', 'slangs_detected',
            'emails_detected', 'num_unique_words'
        ]
    ]

    # Combine TF-IDF features with extracted features
    features_combined = np.hstack((tfidf_features[0], extracted_features))
    features_pca = pca.transform([features_combined])

    return features_pca

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # A simple form for inputting text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        features = preprocess_and_extract_features(text)
        prediction = model.predict(features)
        prediction_text = 'AI-generated' if prediction[0] == 0 else 'Human-written'

        # Generate the explanation
        explainer = LimeTextExplainer(class_names=['AI-generated', 'Human-written'])
        def predict_proba_fn(texts):
            features = np.vstack([preprocess_and_extract_features(text) for text in texts])
            return model.predict_proba(features)

        exp = explainer.explain_instance(text, predict_proba_fn, num_features=10)
        explanation_html = Markup(exp.as_html())

        return render_template('index.html', prediction=prediction_text, explanation=explanation_html, text=text)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
