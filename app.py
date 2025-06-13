from flask import Flask, request, render_template_string
import joblib
import re
import os
import subprocess
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from news2 import run_news

# Ensure models exist
if not all(os.path.exists(p) for p in [
    "news_classifier_model.joblib",
    "tfidf_vectorizer.joblib",
    "label_encoder.joblib"
]):
    print("Required model files not found. Running full pipeline script...")
    run_news()

# ======================
# Load models
model = joblib.load("news_classifier_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# ======================
# Setup Flask
app = Flask(__name__)

# ======================
# HTML Template
HTML_TEMPLATE = """
<!doctype html>
<title>News Category Classifier</title>
<h2>Enter News Article Text to Classify:</h2>
<form method="post" action="/predict">
  <textarea name="text" rows="10" cols="80" placeholder="Paste your article here..."></textarea><br><br>
  <input type="submit" value="Predict Category">
</form>
{% if prediction %}
  <h3>Predicted Category: <span style="color:green">{{ prediction }}</span></h3>
{% endif %}
"""

# ======================
# Text Cleaning
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ======================
# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.form.get("text", "")
    cleaned = clean_text(input_text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)
    label = label_encoder.inverse_transform(pred)[0]
    return render_template_string(HTML_TEMPLATE, prediction=label)

# ======================
# Run
if __name__ == "__main__":
    app.run(debug=True)


