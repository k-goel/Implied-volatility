import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq

# Black-Scholes Formula
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Implied Volatility Solver
def implied_volatility(price, S, K, T, r, option_type="call"):
    objective = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - price
    try:
        return brentq(objective, 1e-6, 5)  # Bounded solver
    except ValueError:
        return np.nan

# Streamlit App
st.title("Implied Volatility Surface Tool")

# Inputs
ticker = st.text_input("Enter Ticker Symbol", "AAPL")
risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0) / 100
dividend_yield = st.slider("Dividend Yield (%)", 0.0, 10.0, 2.0) / 100

if st.button("Calculate"):
    # Fetch data
    stock = yf.Ticker(ticker)
    options = stock.option_chain()  # Replace with loop for all expiries

    # Process data
    call_data = options.calls
    S = stock.history(period="1d")["Close"][-1]  # Current stock price
    T = (pd.to_datetime(call_data['expirationDate']) - pd.Timestamp.now()).dt.days / 365
    K = call_data['strike']
    market_price = call_data['lastPrice']

    # Calculate implied volatility
    call_data['implied_vol'] = [
        implied_volatility(p, S, k, t, risk_free_rate) for p, k, t in zip(market_price, K, T)
    ]

    # 3D Plot
    fig = go.Figure(data=[go.Surface(
        z=call_data.pivot_table(index="strike", columns="expiration", values="implied_vol").values,
        x=call_data['expiration'].unique(),
        y=call_data['strike'].unique()
    )])
    fig.update_layout(title="Implied Volatility Surface", scene=dict(
        xaxis_title="Time to Expiry",
        yaxis_title="Strike Price",
        zaxis_title="Implied Volatility"
    ))
    st.plotly_chart(fig)
