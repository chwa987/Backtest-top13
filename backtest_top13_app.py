import os
os.environ["WATCHDOG_DISABLE"] = "true"

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# =====================
# Hilfsfunktionen
# =====================

@st.cache_data
def load_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Adj Close"]
        return data.dropna(how="all")
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return pd.DataFrame()

def calculate_metrics(prices):
    returns = prices.pct_change().dropna()
    cum_returns = (1 + returns).cumprod()

    cagr = (cum_returns.iloc[-1] ** (252 / len(returns)) - 1).mean()
    volatility = (returns.std() * np.sqrt(252)).mean()
    max_dd = ((cum_returns / cum_returns.cummax()) - 1).min().mean()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)).mean()

    return {
        "CAGR": round(cagr * 100, 2),
        "Volatilität": round(volatility * 100, 2),
        "Max Drawdown": round(max_dd * 100, 2),
        "Sharpe-Ratio": round(sharpe, 2),
    }, cum_returns

# =====================
# Streamlit App
# =====================

st.set_page_config(page_title="📊 Backtest Top 13", layout="wide")
st.title("📈 Backtest – Top 13 mit Exit-Logik")

# 📂 CSV Upload
uploaded_file = st.file_uploader("CSV mit 'Ticker' und optional 'Name' hochladen", type=["csv"])

# 📅 Datumsauswahl
start_date = st.date_input("Startdatum", datetime.date(2018, 1, 1))
end_date = st.date_input("Enddatum", datetime.date.today())

# Umwandeln in datetime.datetime
start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"{len(df)} Ticker aus CSV geladen.")

    if "Ticker" not in df.columns:
        st.error("Die CSV muss mindestens eine Spalte 'Ticker' enthalten.")
    else:
        tickers = df["Ticker"].dropna().unique().tolist()

        if st.button("🚀 Backtest starten"):
            price_data = load_data(tickers, start_date, end_date)

            if not price_data.empty:
                stats, cum_returns = calculate_metrics(price_data)

                # 📊 Chart
                st.line_chart(cum_returns)

                # 📌 Kennzahlen
                st.subheader("Kennzahlen")
                st.json(stats)

                # 📥 CSV Export
                out = cum_returns.reset_index()
                st.download_button(
                    "📥 Ergebnisse als CSV exportieren",
                    out.to_csv(index=False).encode("utf-8"),
                    "backtest_results.csv",
                    "text/csv",
                )
            else:
                st.warning("Keine Kursdaten geladen.")
