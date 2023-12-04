from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import load

app = Flask(__name__)

# Load the trained logistic regression model
logreg = load('trained_model.joblib')
stemmer = PorterStemmer()
cv = load('fitted_vectorizer.joblib')

def preprocess_url(url):
    stemmed_url = ' '.join([stemmer.stem(word) for word in url.split()])
    return cv.transform([stemmed_url])

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    processed_url = preprocess_url(url)
    prediction = logreg.predict(processed_url)[0]
    prediction_label = 'benign' if prediction == 0 else 'malicious'
    return render_template('index.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)