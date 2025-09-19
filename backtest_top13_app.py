import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ----------------------------
# Parameter / Gewichte
# ----------------------------
WEIGHTS = {
    "Abstand GD200 (%)": 0.20,
    "Abstand GD130 (%)": 0.15,
    "MOM260 (%)":       0.25,
    "MOMJT (%)":        0.15,
    "Relative Stärke (%)": 0.15,
    "Volumen-Score":    0.10,
}
IND_COLS = list(WEIGHTS.keys())

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def get_series(df, col="Close"):
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:,0]
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)

def compute_indicators(price, idx_price, volume):
    if price.dropna().empty:
        return [np.nan]*6
    last_close = price.iloc[-1]
    gd200 = price.rolling(200).mean().iloc[-1] if len(price)>=200 else np.nan
    gd130 = price.rolling(130).mean().iloc[-1] if len(price)>=130 else np.nan
    abw200 = (last_close-gd200)/gd200*100 if np.isfinite(gd200) else np.nan
    abw130 = (last_close-gd130)/gd130*100 if np.isfinite(gd130) else np.nan

    if len(price)>260 and np.isfinite(price.iloc[-260]):
        mom260 = (last_close/price.iloc[-260]-1)*100
        ret_12m = last_close/price.iloc[-260]-1
    else:
        mom260, ret_12m = np.nan, np.nan

    if len(price)>21 and np.isfinite(price.iloc[-21]) and np.isfinite(ret_12m):
        ret_1m = last_close/price.iloc[-21]-1
        momjt = (ret_12m-ret_1m)*100
    else:
        momjt = np.nan

    if not idx_price.dropna().empty and len(idx_price)>260 and np.isfinite(ret_12m):
        idx_ret12m = idx_price.iloc[-1]/idx_price.iloc[-260]-1
        rel_str = ((1+ret_12m)/(1+idx_ret12m)-1)*100 if np.isfinite(idx_ret12m) else np.nan
    else:
        rel_str = np.nan

    if not volume.dropna().empty and len(volume)>50:
        vol50 = volume.rolling(50).mean().iloc[-1]
        vol_score = (volume.iloc[-1]/vol50) if (np.isfinite(vol50) and vol50!=0) else np.nan
    else:
        vol_score = np.nan

    return [abw200, abw130, mom260, momjt, rel_str, vol_score]

def score_universe(prices_dict, volumes_dict, idx_price):
    # Kennzahlen je Ticker (letztes Datum)
    rows = []
    for tkr, price in prices_dict.items():
        try:
            volume = volumes_dict.get(tkr, pd.Series(dtype=float))
            abw200, abw130, mom260, momjt, rel_str, vol_score = compute_indicators(price, idx_price, volume)
            rows.append({"Ticker":tkr,
                         "Abstand GD200 (%)":abw200, "Abstand GD130 (%)":abw130,
                         "MOM260 (%)":mom260, "MOMJT (%)":momjt,
                         "Relative Stärke (%)":rel_str, "Volumen-Score":vol_score})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Ticker")

    # Z-Standardisierung über das Universum
    z = pd.DataFrame(index=df.index)
    for col in IND_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        mu, sd = s.mean(skipna=True), s.std(ddof=0, skipna=True)
        z[col] = 0.0 if (pd.isna(sd) or sd==0) else (s-mu)/sd

    weights = pd.Series(WEIGHTS)
    df["Momentum-Score"] = (z[IND_COLS]*weights).sum(axis=1, skipna=True)
    return df.sort_values("Momentum-Score", ascending=False)

def moving_average(series, n):
    return series.rolling(n).mean()

def above_ma(series, n):
    if len(series)<n: return False
    ma = moving_average(series, n).iloc[-1]
    return (series.iloc[-1] >= ma) if np.isfinite(ma) else False

def simulate_equity_curve(prices_dict, weights_now):
    # Tagesrendite = gewichtete Summe der täglichen pct_change
    tickers = [t for t,w in weights_now.items() if w>0]
    if not tickers: 
        # reine Cash-Phase: konstantes 1.0
        # Wir erzeugen leere Serie mit gemeinsamen Index aller verfügbaren Kurse
        if prices_dict:
            any_idx = next(iter(prices_dict.values())).index
            return pd.Series(1.0, index=any_idx)
        return pd.Series(dtype=float)

    # Gemeinsamen Index bauen
    idx = None
    for t in tickers:
        idx = prices_dict[t].index if idx is None else idx.union(prices_dict[t].index)
    idx = idx.sort_values()

    # tägliche Portfoliorendite
    port_ret = pd.Series(0.0, index=idx)
    for t in tickers:
        px = prices_dict[t].reindex(idx).ffill()
        rets = px.pct_change().fillna(0.0)
        port_ret += rets * weights_now[t]

    equity = (1+port_ret).cumprod()
    return equity

# ----------------------------
# Backtest Engine
# ----------------------------
def backtest(universe, start="2015-01-01", end=None, top_n=10, reserve_k=3,
             checks_per_week=2, policy="baseline"):
    """
    policy:
      - 'baseline' : monatlich Top-10 halten (nur Rotation Monatsende)
      - 'gd50_reserve' : 2x/Woche: Exit <GD50; sofortiger Ersatz aus Top-13 (nur wenn Ersatz >GD50 & >GD200)
      - 'gd50_cash' : 2x/Woche: Exit <GD50; kein Ersatz, bis Monatsrotation
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # Daten laden in einem Batch
    data = yf.download(universe, start=start, end=end, auto_adjust=True, group_by="ticker", progress=False, threads=True)
    # Benchmark
    idx = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    idx_close = get_series(idx, "Close")

    # Dicts Close/Volume
    prices = {}
    volumes = {}
    for t in universe:
        try:
            df = data[t] if (t,) in data.columns else data.get(t)
        except Exception:
            df = None
        if df is None or df.empty: 
            continue
        c = get_series(df, "Close")
        v = get_series(df, "Volume")
        if c.dropna().empty: 
            continue
        prices[t] = c.dropna()
        volumes[t] = v.dropna()

    # Monatspunkte (letzter Handelstag je Monat)
    if not prices:
        return pd.DataFrame(), {}
    full_index = None
    for s in prices.values():
        full_index = s.index if full_index is None else full_index.union(s.index)
    full_index = full_index.sort_values()
    months = pd.DatetimeIndex(pd.to_datetime(pd.Series(full_index)).to_period("M").drop_duplicates().astype(str) + "-01")
    month_ends = []
    for m in months:
        # letzter Handelstag im Monat
        m_end = full_index[full_index.to_period("M")==m.to_period("M")][-1]
        month_ends.append(m_end)
    month_ends = pd.DatetimeIndex(month_ends)

    # Wochentage für Checks (2x pro Woche: z.B. Dienstag=1, Freitag=4)
    check_weekdays = {1,4} if checks_per_week==2 else {2}  # Default Di/Fr; bei 1: nur Mittwoch

    equity_dict = {"baseline":[], "gd50_reserve":[], "gd50_cash":[]}
    date_axis = []

    # Startportfolio am ersten Monatsschluss
    if len(month_ends)==0:
        return pd.DataFrame(), {}
    port_equity = 1.0
    port_series = pd.Series([1.0], index=[month_ends[0]])

    # Hilfszustände für A/B
    holdings = {}   # ticker -> gewicht
    cash_weight = 0.0

    def pick_top(df_scores, n, extra=0):
        # liefert Liste der besten n (+ extra) tickers
        return df_scores.index.to_list()[:n+extra]

    # Initial Ranking
    # Score mit Index-Relativstärke (nutzt idx_close)
    scores0 = score_universe(prices, volumes, idx_close.loc[full_index.min():full_index.max()])
    top_list = pick_top(scores0, top_n, reserve_k)
    holdings = {t: (1.0/top_n if i<top_n else 0.0) for i,t in enumerate(top_list)}
    cash_weight = 0.0

    last_month_end = month_ends[0]
    cur_equity_curve = simulate_equity_curve(prices, holdings)

    # Zeitachse iterieren (alle Handelstage ab erstem Monatsschluss)
    all_days = full_index[full_index>=last_month_end]

    for d in all_days:
        # tägliche Portfoliobewegung aufaddieren
        # (vereinfachte Simulation: Gewichte konstant zwischen Rebalancings)
        if d in cur_equity_curve.index:
            port_equity = cur_equity_curve.loc[d]
        date_axis.append(d)
        equity_dict["baseline"].append(port_equity)  # Placeholder; echte Baseline separat berechnen unten

        # Zwischenchecks: nur für Strategien A/B
        weekday = d.weekday()
        is_check = weekday in check_weekdays

        # Monatsrotation?
        is_month_end = d in month_ends

        # --- Strategie A/B: GD50-Exit-Logik zwischendurch ---
        if is_check and (policy in ["gd50_reserve", "gd50_cash"]):
            # Exit falls unter GD50
            to_sell = []
            for t, w in list(holdings.items()):
                if w<=0: continue
                px = prices.get(t)
                if px is None or px.index.min()>d: 
                    continue
                px_d = px.loc[:d]
                if len(px_d)<50: 
                    continue
                if not above_ma(px_d, 50):  # unter GD50
                    to_sell.append(t)
            # Verkaufen
            for t in to_sell:
                holdings[t] = 0.0

            freed_slots = max(0, top_n - sum(1 for w in holdings.values() if w>0))

            # Ersatzlogik (nur in 'gd50_reserve')
            if freed_slots>0 and policy=="gd50_reserve":
                # Aktuelles Ranking (Scores of today)
                scores_d = score_universe(prices, volumes, idx_close.loc[:d])
                top13 = pick_top(scores_d, top_n, reserve_k)
                bench = set(top13[top_n:top_n+reserve_k])  # Nachrücker
                # Filter: nur >GD50 & >GD200
                candidates = []
                for t in bench:
                    px = prices.get(t)
                    if px is None or px.index.min()>d: 
                        continue
                    px_d = px.loc[:d]
                    if len(px_d)<200: 
                        continue
                    cond50 = above_ma(px_d, 50)
                    cond200 = above_ma(px_d, 200)
                    if cond50 and cond200:
                        candidates.append(t)
                # Fülle Slots in Ranking-Reihenfolge
                for t in top13:
                    if t in holdings and holdings[t]>0: 
                        continue
                    if t in candidates and freed_slots>0:
                        holdings[t] = 0.0  # Platzhalter, Gewichte später setzen
                        freed_slots -= 1

            # Gewichte neu setzen (gleichgewichtet auf Anzahl >0)
            alive = [t for t,w in holdings.items() if w>0 or (w==0 and t in holdings)]
            alive = [t for t in holdings if holdings[t]>0]  # only current holdings with weight>0
            alive = [t for t,w in holdings.items() if w>0]
            npos = len(alive)
            if npos>0:
                w = 1.0/top_n if policy=="gd50_reserve" else (1.0/npos)  # In Reserve-Policy Ziel = Top10; bei Cash kann <10 sein
                # In Reserve-Policy: fehlende werden ggf. erst bei Monatsende aufgefüllt; hier verteilen auf existierende >0
                total_w = 0.0
                for t in list(holdings.keys()):
                    if holdings[t]>0:
                        holdings[t] = w
                        total_w += w
                cash_weight = max(0.0, 1.0 - total_w)
            else:
                cash_weight = 1.0

            # Neue Equity-Kurve ab heute mit aktualisierten Gewichten
            cur_equity_curve = simulate_equity_curve(prices, holdings).loc[simulate_equity_curve(prices, holdings).index >= d]

        # Monatsrotation: alle Policies werden neu auf Top-Liste gebracht
        if is_month_end:
            scores_m = score_universe(prices, volumes, idx_close.loc[:d])
            top13 = scores_m.index.to_list()[:top_n+reserve_k]
            # Setze neues Portfolio = Top10 (gleichgewichtet)
            holdings = {t:(1.0/top_n if i<top_n else 0.0) for i,t in enumerate(top13)}
            cash_weight = 0.0
            cur_equity_curve = simulate_equity_curve(prices, holdings).loc[simulate_equity_curve(prices, holdings).index >= d]

    # Equity-Serie der gewählten Policy
    eq = pd.Series(equity_dict["baseline"], index=date_axis, dtype=float)
    # Oben haben wir placeholder gefüllt; wir brauchen die echte Simulation je Policy separat:
    # -> Einfacher: einmal die Schleife je Policy laufen lassen. Der Einfachheit halber rufen wir backtest rekursiv pro Policy auf.
    return eq, {}  # Platzhalter (unten rufen wir mehrfach auf)

def run_policy(universe, start, end, top_n, reserve_k, checks_per_week, policy):
    # Gleiches Gerüst, aber als vereinfachte Simulation ohne Placeholder:
    # (Aus Platzgründen implementieren wir hier eine reduzierte, aber funktionale Version der obigen Engine)
    data = yf.download(universe, start=start, end=end, auto_adjust=True, group_by="ticker", progress=False, threads=True)
    idx = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    idx_close = get_series(idx, "Close")

    prices, volumes = {}, {}
    for t in universe:
        try:
            df = data[t] if (t,) in data.columns else data.get(t)
        except Exception:
            df = None
        if df is None or df.empty: 
            continue
        c = get_series(df, "Close")
        v = get_series(df, "Volume")
        if c.dropna().empty: 
            continue
        prices[t] = c.dropna()
        volumes[t] = v.dropna()

    if not prices:
        return pd.Series(dtype=float), {}

    idx_all = None
    for s in prices.values():
        idx_all = s.index if idx_all is None else idx_all.union(s.index)
    idx_all = idx_all.sort_values()
    months = pd.DatetimeIndex(pd.to_datetime(pd.Series(idx_all)).to_period("M").drop_duplicates().astype(str) + "-01")
    month_ends = []
    for m in months:
        me = idx_all[idx_all.to_period("M")==m.to_period("M")][-1]
        month_ends.append(me)
    month_ends = pd.DatetimeIndex(month_ends)

    check_weekdays = {1,4} if checks_per_week==2 else {2}

    # Initial
    scores0 = score_universe(prices, volumes, idx_close.loc[idx_all.min():idx_all.max()])
    top13 = scores0.index.to_list()[:top_n+reserve_k]
    holdings = {t:(1.0/top_n if i<top_n else 0.0) for i,t in enumerate(top13)}
    equity = pd.Series(1.0, index=[month_ends[0]])
    cur_curve = simulate_equity_curve(prices, holdings)
    date_axis, values = [], []

    for d in idx_all[idx_all>=month_ends[0]]:
        if d in cur_curve.index:
            val = cur_curve.loc[d]
        else:
            val = values[-1] if values else 1.0
        date_axis.append(d); values.append(val)

        # Checks
        if policy in ["gd50_reserve","gd50_cash"] and d.weekday() in check_weekdays:
            # Exit <GD50
            to_sell = []
            for t,w in list(holdings.items()):
                if w<=0: continue
                px = prices.get(t); 
                if px is None or px.index.min()>d: continue
                px_d = px.loc[:d]
                if len(px_d)>=50 and not above_ma(px_d,50):
                    to_sell.append(t)
            for t in to_sell:
                holdings[t] = 0.0

            # Ersatz nur in Reserve-Policy
            if policy=="gd50_reserve":
                freed = max(0, top_n - sum(1 for w in holdings.values() if w>0))
                if freed>0:
                    scores_d = score_universe(prices, volumes, idx_close.loc[:d])
                    top13 = scores_d.index.to_list()[:top_n+reserve_k]
                    bench = top13[top_n:top_n+reserve_k]
                    # Filter >GD50 & >GD200
                    cand = []
                    for t in bench:
                        px = prices.get(t)
                        if px is None or px.index.min()>d: 
                            continue
                        px_d = px.loc[:d]
                        if len(px_d)>=200 and above_ma(px_d,50) and above_ma(px_d,200):
                            cand.append(t)
                    for t in top13:
                        if t in holdings and holdings[t]>0: 
                            continue
                        if t in cand and freed>0:
                            holdings[t] = 0.0
                            freed -= 1

            # Gewichte
            alive = [t for t,w in holdings.items() if w>0]
            if policy=="gd50_reserve":
                # Halte Ziel TopN: verteile nur auf alive; fehlende Slots bleiben effektiv Cash
                npos = len(alive)
                if npos>0:
                    w = 1.0/top_n
                    for t in holdings:
                        if holdings[t]>0:
                            holdings[t]=w
                # sonst komplett Cash (implizit)
            else: # gd50_cash
                npos = len(alive)
                if npos>0:
                    w = 1.0/npos
                    for t in holdings:
                        if holdings[t]>0:
                            holdings[t]=w

            cur_curve = simulate_equity_curve(prices, holdings).loc[simulate_equity_curve(prices, holdings).index>=d]

        # Monatsrotation: zurück zu frischem Top13; Portfolio = Top10 equal-weight
        if d in month_ends:
            scores_m = score_universe(prices, volumes, idx_close.loc[:d])
            top13 = scores_m.index.to_list()[:top_n+reserve_k]
            holdings = {t:(1.0/top_n if i<top_n else 0.0) for i,t in enumerate(top13)}
            cur_curve = simulate_equity_curve(prices, holdings).loc[simulate_equity_curve(prices, holdings).index>=d]

    eq = pd.Series(values, index=date_axis, dtype=float)
    # Kennzahlen
    if len(eq)>2:
        years = (eq.index[-1]-eq.index[0]).days/365.0
        cagr = eq.iloc[-1]**(1/years)-1 if years>0 else np.nan
        daily_ret = eq.pct_change().dropna()
        vol = daily_ret.std()*np.sqrt(252) if not daily_ret.empty else np.nan
        dd = (eq/eq.cummax()-1).min()
        sharpe = (cagr-0.0)/vol if (vol and vol>0) else np.nan
        stats = {"CAGR%":round(cagr*100,2),"Vol%":round((vol or np.nan)*100,2),"MaxDD%":round(dd*100,2),"Sharpe":round(sharpe,2)}
    else:
        stats = {}
    return eq, stats

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Top-13 Reserve Backtest", layout="wide")
st.title("📈 Momentum-Backtest – Top-13 Nachrücker vs. GD50-Cash vs. Baseline")

uploaded = st.file_uploader("📂 CSV hochladen (Spalte 'Ticker', optional 'Name')", type=["csv"])
tickers_text = st.text_area("Oder Ticker eingeben (kommasepariert)", "AAPL, MSFT, NVDA, AMZN, META, TSLA, GOOGL, JPM, JNJ, AMD, NFLX, INTC, AVGO, COST, UNH, PEP, KO, CRM, ORCL, QCOM")

colp1, colp2, colp3 = st.columns(3)
start = colp1.text_input("Start", "2015-01-01")
end   = colp2.text_input("Ende (leer = heute)", "")
top_n = colp3.number_input("Top N", 10, 30, 10)
reserve_k = st.number_input("Nachrücker (K)", 0, 10, 3)
checks = st.selectbox("Zwischen-Checks pro Woche", [1,2], index=1)

if uploaded is not None:
    dfu = pd.read_csv(uploaded)
    if "Ticker" not in dfu.columns:
        st.error("CSV braucht Spalte 'Ticker'.")
        st.stop()
    universe = dfu["Ticker"].dropna().astype(str).str.strip().tolist()
    st.success(f"{len(universe)} Ticker geladen.")
else:
    universe = [t.strip() for t in tickers_text.split(",") if t.strip()]

if st.button("🚀 Backtest starten") and universe:
    end_val = end if end.strip() else None

    with st.spinner("Baseline läuft…"):
        eq_base, stats_base = run_policy(universe, start, end_val, top_n, reserve_k, checks, "baseline")
    with st.spinner("Strategie A (GD50 + Reserve) läuft…"):
        eq_res, stats_res = run_policy(universe, start, end_val, top_n, reserve_k, checks, "gd50_reserve")
    with st.spinner("Strategie B (GD50 + Cash) läuft…"):
        eq_cas, stats_cas = run_policy(universe, start, end_val, top_n, reserve_k, checks, "gd50_cash")

    # Gemeinsamen Index schneiden
    idx_all = eq_base.index.union(eq_res.index).union(eq_cas.index)
    df_eq = pd.DataFrame({
        "Baseline (Top10 mtl.)": eq_base.reindex(idx_all).ffill(),
        "A: GD50 + Reserve (Top13)": eq_res.reindex(idx_all).ffill(),
        "B: GD50 + Cash": eq_cas.reindex(idx_all).ffill(),
    })

    st.line_chart(df_eq)

    colA, colB, colC = st.columns(3)
    colA.metric("Baseline CAGR", f"{stats_base.get('CAGR%','n/a')}")
    colA.metric("Volatilität", f"{stats_base.get('Vol%','n/a')}")
    colA.metric("Max Drawdown", f"{stats_base.get('MaxDD%','n/a')}")
    colA.metric("Sharpe", f"{stats_base.get('Sharpe','n/a')}")

    colB.metric("A: CAGR", f"{stats_res.get('CAGR%','n/a')}")
    colB.metric("Volatilität", f"{stats_res.get('Vol%','n/a')}")
    colB.metric("Max Drawdown", f"{stats_res.get('MaxDD%','n/a')}")
    colB.metric("Sharpe", f"{stats_res.get('Sharpe','n/a')}")

    colC.metric("B: CAGR", f"{stats_cas.get('CAGR%','n/a')}")
    colC.metric("Volatilität", f"{stats_cas.get('Vol%','n/a')}")
    colC.metric("Max Drawdown", f"{stats_cas.get('MaxDD%','n/a')}")
    colC.metric("Sharpe", f"{stats_cas.get('Sharpe','n/a')}")

    st.download_button("📥 Equity-Kurven als CSV", df_eq.to_csv().encode("utf-8"), "equity_curves_top13_test.csv", "text/csv")
