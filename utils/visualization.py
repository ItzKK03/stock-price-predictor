import streamlit as st
import matplotlib.pyplot as plt

def plot_technical_indicators(df):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(df['Close'], label='Close Price', color='blue')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    axs[1].plot(df['MA20'], label='MA20', color='green')
    axs[1].plot(df['MA50'], label='MA50', color='red')
    axs[1].set_ylabel('Moving Averages')
    axs[1].legend()

    axs[2].plot(df['RSI'], label='RSI', color='purple')
    axs[2].set_ylabel('RSI')
    axs[2].axhline(70, color='r', linestyle='--')
    axs[2].axhline(30, color='g', linestyle='--')
    axs[2].legend()

    plt.tight_layout()
    st.pyplot(fig)
