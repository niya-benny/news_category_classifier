
# 📰 News Category Classifier (End-to-End NLP Pipeline)

This project implements a **complete end-to-end News Category Classification system**, starting from **web scraping live news articles** to **automatic labeling using a transformer model**, followed by **data cleaning, EDA, classical ML model training**, and finally **deployment using a Flask web application**.

---

## 📌 Project Highlights

✔ Web scraping news articles from multiple real-world sources
✔ Automatic labeling using a **pretrained BERT news classification model**
✔ Text preprocessing using **NLTK**
✔ Exploratory Data Analysis (EDA) with visualizations
✔ Feature extraction using **TF-IDF**
✔ Classification using **Logistic Regression**
✔ Model evaluation with **classification report & confusion matrix**
✔ Deployed as a **Flask web application**

---

## 🧠 Overall Architecture

```
Web Scraping
     ↓
Transformer-based Auto Labeling (BERT)
     ↓
Text Cleaning & Preprocessing (NLTK)
     ↓
EDA & Visualization
     ↓
TF-IDF Feature Extraction
     ↓
Logistic Regression Training
     ↓
Model Evaluation
     ↓
Flask Web App Deployment
```

---
## 🛠️ Technologies Used

* **Python**
* **Flask**
* **NLTK**
* **Scikit-learn**
* **Transformers (HuggingFace)**
* **Newspaper3k**
* **Matplotlib & Seaborn**
* **WordCloud**

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/niya-benny/news_category_classifier.git
cd news_category_classifier
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk flask transformers torch newspaper3k matplotlib seaborn wordcloud joblib
```

---

## ▶️ Running the Project

### 🔹 Step 1: Run Full Pipeline (Scraping → Training)

```bash
python news2.py
```

This will:

* Scrape news articles
* Auto-label them using a BERT model
* Clean and preprocess text
* Perform EDA
* Train and evaluate the classifier
* Save trained models

> ⚠️ This step may take time due to web scraping and transformer inference.

---

### 🔹 Step 2: Launch Flask Web App

```bash
python app.py
```

Open browser and go to:

```
http://127.0.0.1:5000/
```

Paste any news article text to get its **predicted category**.

---

## 🧪 Model Details

* **Auto-labeling Model:**
  `elozano/bert-base-cased-news-category`

* **Feature Extraction:**
  TF-IDF (max 5000 features)

* **Classifier:**
  Logistic Regression (class-balanced)

* **Evaluation Metrics:**
  Precision, Recall, F1-Score, Confusion Matrix

---

## 📊 Exploratory Data Analysis (EDA)

Generated outputs include:

* Category distribution
* Article length distribution
* Word clouds for top 5 categories
* Dataset summary report

All EDA results are saved in:

```
plots/
EDA_summary.txt
```

---

## 🎯 Use Cases

* NLP & Text Classification learning
* End-to-end ML pipeline demonstration
* Academic mini/major project
* Resume and internship showcase
* News analytics systems

---

## 🚀 Future Enhancements

* Replace Logistic Regression with **BERT fine-tuning**
* Add REST API support
* Deploy on **AWS / Render / Hugging Face Spaces**
* Add real-time scraping and prediction
* Improve UI using HTML/CSS templates

---


