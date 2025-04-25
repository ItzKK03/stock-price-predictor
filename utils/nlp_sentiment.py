# utils/nlp_sentiment.py

from textblob import TextBlob

def analyze_sentiment(headlines):
    sentiments = []
    labels = []
    
    for text in headlines:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiments.append(polarity)
        if polarity > 0.1:
            labels.append("Positive")
        elif polarity < -0.1:
            labels.append("Negative")
        else:
            labels.append("Neutral")
    
    return sentiments, labels
