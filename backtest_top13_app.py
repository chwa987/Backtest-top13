import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.title("📊 Backtest – Momentum Top20 mit Top10+Reserve")

# CSV Upload
uploaded_file = st.file_uploader("CSV mit 'Ticker' und optional 'Name' hochladen", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    tickers = df["Ticker"].dropna().unique().tolist()
else:
    st.stop()

# Datumswahl
start_date = st.date_input("Startdatum", datetime(2018,1,1))
end_date = st.date_input("Enddatum", datetime.today())

# Kursdaten laden
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        raise KeyError("Weder Adj Close noch Close in Yahoo Finance Daten!")
    return prices.dropna(how="all")

prices = load_data(tickers, start_date, end_date)

# GD50 Berechnung
gd50 = prices.rolling(50).mean()

# Momentum-Berechnung (z. B. 6M = 126 Tage)
momentum = prices.pct_change(126)

# Backtest
def run_backtest(prices, gd50, momentum, top_n=20, active_n=10, reserve_n=3, rebalance_freq="M"):
    portfolio = pd.Series(dtype=float)
    cash = 1.0  # Startwert 1.0 = 100%
    
    for date, _ in prices.resample(rebalance_freq).first().iterrows():
        if date not in prices.index: 
            continue

        # Ranking zum Stichtag
        rank = momentum.loc[date].dropna().sort_values(ascending=False).head(top_n)

        active, reserve = [], []
        for t in rank.index:
            price_today = prices.loc[date, t]
            ma_today = gd50.loc[date, t]
            if pd.isna(price_today) or pd.isna(ma_today):
                continue
            if price_today > ma_today:
                if len(active) < active_n:
                    active.append(t)
                elif len(reserve) < reserve_n:
                    reserve.append(t)

        selection = active + reserve
        if not selection:
            continue

        # Portfolio-Performance
        returns = prices[selection].pct_change().loc[date:].iloc[0]
        cash *= (1 + returns.mean())  # Gleichgewichtung
        portfolio.loc[date] = cash

    return portfolio

# Ausführen
if st.button("🚀 Backtest starten"):
    result = run_backtest(prices, gd50, momentum)
    if not result.empty:
        # Plot
        st.line_chart(result)

        # Kennzahlen
        cagr = (result.iloc[-1]) ** (252/len(result)) - 1
        vol = result.pct_change().std() * np.sqrt(252)
        mdd = ((result / result.cummax()) - 1).min()
        sharpe = cagr / vol if vol != 0 else np.nan

        st.json({
            "CAGR": round(cagr*100,2),
            "Volatilität": round(vol*100,2),
            "Max Drawdown": round(mdd*100,2),
            "Sharpe-Ratio": round(sharpe,2)
        })

        # Export
        csv = result.to_frame("Portfolio").to_csv().encode("utf-8")
        st.download_button("📥 Ergebnisse als CSV exportieren", csv, "backtest_results.csv")
