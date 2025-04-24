from transformers import pipeline
import requests

def fetch_news_headlines(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey=475f89bb10104be68933b0974559b5d4"
    response = requests.get(url)
    data = response.json()
    headlines = [article['title'] for article in data.get('articles', [])][:10]
    return headlines

def analyze_sentiment(headlines):
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = classifier(headlines)
    
    sentiment_scores = []
    labels = []
    
    for result in results:
        label = result['label']
        score = result['score']
        labels.append(label)
        sentiment_scores.append(score if label == 'POSITIVE' else -score)

    return sentiment_scores, labels
