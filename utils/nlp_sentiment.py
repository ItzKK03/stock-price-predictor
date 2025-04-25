from textblob import TextBlob

def analyze_sentiment(headlines):
    sentiment_scores = []
    labels = []

    for headline in headlines:
        blob = TextBlob(headline)
        score = blob.sentiment.polarity
        sentiment_scores.append(score)
        if score > 0.1:
            labels.append("POSITIVE")
        elif score < -0.1:
            labels.append("NEGATIVE")
        else:
            labels.append("NEUTRAL")

    return sentiment_scores, labels
