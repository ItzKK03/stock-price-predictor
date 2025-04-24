# 📈 Stock Price Predictor

An advanced **AI-powered web application** that predicts stock closing prices using **LSTM deep learning**, **technical indicators**, and **NLP-based sentiment analysis** from news headlines. Built with a modern and interactive **Streamlit UI**.

---

## 🚀 Features

- 📊 **LSTM-Based Stock Price Prediction**
- 🧠 **Sentiment Analysis from News (BERT + Transformers)**
- 📉 **Technical Indicators: RSI, MA20, MA50, Volume**
- 📈 **Interactive Charts using Matplotlib & Streamlit**
- 🧩 Modular structure for scalability
- 🛠️ Easy to extend, optimize, and deploy

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy / Matplotlib
- Yahoo Finance API
- Hugging Face Transformers (BERT)
- Streamlit (Frontend)
- Joblib (Model persistence)

---

## 📂 Folder Structure

```
stock-price-predictor/
├── app.py                        # Streamlit Web App
├── predict.py                   # Prediction pipeline
├── train.py                     # Training wrapper
├── requirements.txt
├── model/
│   └── train_lstm.py
├── saved_models/
│   └── lstm_stock_model.h5
│   └── scaler.save
├── utils/
│   ├── indicators.py
│   ├── visualization.py
│   └── nlp_sentiment.py
├── .streamlit/
│   └── config.toml
```

---

## 💻 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/ItzKK03/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Access on `http://localhost:8501`

---

## 📌 TODO / Upcoming Features

- Paper trading integration
- Backtesting module
- Real-time alerts
- Improved performance with GRU/Transformer models

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify!

---

## 🙌 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/)
- [Hugging Face Transformers](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- Built with 💙 by **Kartikay Kant**

---

## 💬 Contact

**LinkedIn:** [linkedin.com/in/kartik-kant](https://linkedin.com/in/kartik-kant)  
**GitHub:** [@ItzKK03](https://github.com/ItzKK03)
