import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

st.set_page_config(page_title="📊 Backtest Top 13", layout="wide")

st.title("📊 Momentum Backtest – Top 13 Strategie")

# CSV Upload
uploaded_file = st.file_uploader("CSV mit 'Ticker' und optional 'Name' hochladen", type=["csv"])
if uploaded_file is not None:
    universe = pd.read_csv(uploaded_file)
    st.success(f"{len(universe)} Ticker aus CSV geladen.")
else:
    st.info("Bitte CSV hochladen, z. B. champions_with_names.csv")
    st.stop()

# Start- und Enddatum
start_date = st.date_input("Startdatum", datetime.date(2018, 1, 1))
end_date = st.date_input("Enddatum", datetime.date.today())

# Parameter
top_n = st.number_input("Anzahl Top Aktien", min_value=5, max_value=20, value=10)
reserve_k = st.number_input("Reserve (z. B. bis 13)", min_value=0, max_value=10, value=3)

# Daten laden
tickers = universe["Ticker"].tolist()
names = universe["Name"].tolist() if "Name" in universe.columns else tickers

st.write("Tickers:", tickers[:10], "…")

@st.cache_data
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    return df

price_data = load_data(tickers, start_date, end_date)

# Monatsrenditen berechnen
returns = price_data.pct_change().dropna()

# -------------------------------
# Robuste run_policy Funktion
# -------------------------------
def run_policy(universe, start, end, top_n, reserve_k, checks, label="base"):
    results = {}
    portfolio = []

    idx_all = universe.index
    if not isinstance(idx_all, pd.DatetimeIndex):
        idx_all = pd.to_datetime(idx_all, errors="coerce")

    months = (
        pd.Series(idx_all)
        .dropna()
        .dt.to_period("M")
        .drop_duplicates()
        .sort_values()
        .dt.to_timestamp()
    )

    equity_curve = []
    for month in months:
        try:
            # Momentum = Performance letzte 6 Monate
            momentum = universe.loc[:month].pct_change().tail(126).mean()
            ranked = momentum.sort_values(ascending=False).head(top_n + reserve_k)

            picks = ranked.index[:top_n]
            portfolio = list(picks)

            monthly_ret = universe.loc[month, portfolio].pct_change().mean()
            equity_curve.append(monthly_ret)
        except Exception as e:
            print(f"[{label}] Fehler bei {month}: {e}")
            equity_curve.append(0)

    equity_curve = pd.Series(equity_curve, index=months)
    results["Portfolio"] = equity_curve

    stats = {
        "CAGR": (1 + equity_curve).prod() ** (12 / len(equity_curve)) - 1,
        "Volatilität": equity_curve.std() * np.sqrt(12),
        "Max Drawdown": (equity_curve.cummax() - equity_curve).max(),
        "Sharpe-Ratio": equity_curve.mean() / equity_curve.std() if equity_curve.std() > 0 else 0,
    }

    return results, stats

# Backtest starten
if st.button("🚀 Backtest starten"):
    eq_base, stats_base = run_policy(price_data, start_date, end_date, top_n, reserve_k, checks=True, label="base")

    st.line_chart(eq_base["Portfolio"].cumprod())

    st.subheader("📊 Kennzahlen")
    st.json(stats_base)

    st.download_button("📥 Ergebnisse als CSV", eq_base["Portfolio"].to_csv().encode("utf-8"), "backtest_results.csv", "text/csv")
