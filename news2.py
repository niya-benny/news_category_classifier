import os
import pandas as pd
from newspaper import Article, build
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

#=======================================================================
#Process 1: Web scraping and labeling the web scraped dataset using a transformer model

def run_news():
    labeled_file = "labeled_news_articles.csv"

    # Check if labeled dataset already exists
    if not os.path.exists(labeled_file):

        print("Labeled dataset not found. Starting scraping and labeling...")

        # News sources to scrape
        news_sources = {
            "BBC": "https://www.bbc.com/news",
            "Indian Express": "https://indianexpress.com/section/india/",
            "Times of India": "https://timesofindia.indiatimes.com/india",
            "NBC": "https://www.nbcnews.com/",
            "HuffPost": "https://www.huffpost.com/",
            "National Geographic": "https://www.nationalgeographic.com/",
            "Medical News Today": "https://www.medicalnewstoday.com/",
            "The Verge": "https://www.theverge.com/",
            "TechCrunch": "https://techcrunch.com/",
            "Hindustan Times": "https://www.hindustantimes.com/",
            "Science Daily": "https://www.sciencedaily.com/",
            "CNN": "https://edition.cnn.com/"
        }

        data = []

        def scrape_articles(source_url, max_articles=10):
            try:
                paper = build(source_url, memoize_articles=False)
                print(f"\nScraping from: {source_url} | Found {len(paper.articles)} articles")

                count = 0
                for article in paper.articles:
                    if count >= max_articles:
                        break
                    try:
                        article.download()
                        article.parse()
                        if article.text.strip():
                            data.append({
                                "url": article.url,
                                "content": article.text.strip()
                            })
                            count += 1
                    except Exception as e:
                        print(f"Error processing article: {e}")
                    time.sleep(1)
            except Exception as e:
                print(f"Error accessing source {source_url}: {e}")

        # Scrape articles
        for name, url in news_sources.items():
            scrape_articles(url, max_articles=100)

        df = pd.DataFrame(data)
        print(f"\nTotal articles scraped: {len(df)}")
        df.to_csv("news_articles.csv", index=False)
        print("Saved scraped articles to 'news_articles.csv'")

        # Load model and tokenizer
        model_name = "elozano/bert-base-cased-news-category"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        id2label = model.config.id2label

        # Label prediction function
        def predict_label(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            predicted_class_id = outputs.logits.argmax().item()
            return id2label[predicted_class_id]

        # Labeling content
        df['label'] = df['content'].apply(lambda x: predict_label(str(x)[:512]))
        df.to_csv(labeled_file, index=False)
        print(f"Labeled data saved to '{labeled_file}'")

    else:
        print(f"'{labeled_file}' already exists. Skipping scraping and labeling.")

    #===============================================================================
    #Process 2:Clean and preprocess the collected data.

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Load the labeled dataset
    df = pd.read_csv("labeled_news_articles.csv")

    # Initialize tools
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()  # Lowercase
        text = re.sub(r'[^a-z\s]', ' ', text)  # Remove punctuation/numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    # Apply cleaning
    df['cleaned_content'] = df['content'].astype(str).apply(clean_text)

    # Save preprocessed dataset
    df.to_csv("cleaned_labeled_news_articles.csv", index=False)
    print("Cleaned and preprocessed data saved to 'cleaned_labeled_news_articles.csv'")

    #========================================================================

    #Process 3: Perform exploratory data analysis (EDA) to understand the dataset

    # Ensure plot directory exists
    os.makedirs("plots", exist_ok=True)

    # Load cleaned and labeled data
    df = pd.read_csv("cleaned_labeled_news_articles.csv")

    # Create and open a text file to save info
    with open("EDA_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Dataset Info ===\n")
        df.info(buf=f)

        f.write("\n\n=== Sample Rows ===\n")
        f.write(df.head().to_string())

        f.write("\n\n=== Category Distribution ===\n")
        f.write(df['label'].value_counts().to_string())

    # Plot category distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(y='label', data=df, order=df['label'].value_counts().index, palette='viridis')
    plt.title("News Category Distribution")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig("plots/category_distribution.png")
    plt.close()

    # Article length analysis
    df['word_count'] = df['cleaned_content'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(df['word_count'], bins=30, kde=True, color='skyblue')
    plt.title("Distribution of Article Length (Words)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/article_length_distribution.png")
    plt.close()

    # WordCloud for top 5 categories
    top_categories = df['label'].value_counts().index[:5]

    for cat in top_categories:
        text = " ".join(df[df['label'] == cat]['cleaned_content'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud for '{cat}' Category")
        plt.tight_layout()
        plt.savefig(f"plots/wordcloud_{cat.replace(' ', '_')}.png")
        plt.close()

    print("EDA completed. Results saved in 'EDA_summary.txt' and 'plots/' directory.")

    #===================================================================

    #Process 4: Model Building

    # Load data
    df = pd.read_csv("cleaned_labeled_news_articles.csv")

    # Drop rows with missing text or label
    df.dropna(subset=["cleaned_content", "label"], inplace=True)

    # Encode labels
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_content"], df["label_encoded"], test_size=0.2, random_state=42, stratify=df["label_encoded"]
    )

    # Convert text to TF-IDF vectors(NLP Technique to transform raw text)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test_tfidf)

    os.makedirs("evaluation_reports", exist_ok=True)

    # === Save classification report to text file ===
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    with open("evaluation_reports/classification_report.txt", "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(report)

    # === Save confusion matrix as image ===
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("evaluation_reports/confusion_matrix.png")
    plt.close()

    print("Evaluation reports saved to 'evaluation_reports/' folder.")

    # Save the model and transformers
    joblib.dump(clf, "news_classifier_model.joblib")
    joblib.dump(tfidf, "tfidf_vectorizer.joblib")
    joblib.dump(le, "label_encoder.joblib")

    print("\nModel and vectorizer saved successfully.")

    #================================================================

if __name__ == "__main__":
    run_news()
