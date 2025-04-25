import streamlit as st
from predict import predict_next_day_close
from utils.visualization import plot_technical_indicators
from utils.nlp_sentiment import analyze_sentiment

st.set_page_config(layout="wide", page_title="AI Stock Predictor")

st.title("📈 AI-Powered Stock Price Predictor")
ticker = st.text_input("Enter Ticker (e.g. AAPL)", "TSLA", "AAPL")

if st.button("Run Prediction"):
    with st.spinner("Fetching Data & Predicting..."):
        df, predicted_close = predict_next_day_close(ticker)
        st.success(f"📉 Predicted Closing Price for Tomorrow: **${predicted_close:.2f}**")

        st.subheader("📊 Technical Indicators")
        plot_technical_indicators(df)

        st.subheader("📰 News Sentiment Analysis")
        sentiments, _ = analyze_sentiment(headlines)

        sentiment_score = sum([int(s[0]) for s in sentiments if s[0].isdigit()]) / len(sentiments)
        st.write("Sentiment Score (1-5):", sentiment_score)
        for headline, sentiment in zip(headlines, sentiments):
            st.markdown(f"- **{headline}** → *{sentiment}*")

st.markdown("---")
st.caption("Made with ❤️ by Kartikay Kant | Powered by LSTM + BERT + Streamlit")
