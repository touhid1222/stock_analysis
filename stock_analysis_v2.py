# -*- coding: utf-8 -*-
##############################################################################################
# Stock Decision Dashboard — buy low / sell high helper (not financial advice)
# streamlit run app.py
##############################################################################################

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Decision Dashboard", layout="wide")

# ==============================
# Header
# ==============================
st.title("📈 Stock Decision Dashboard")
st.caption("Goal: find good **buy** setups near local lows and **sell/trim** near local highs. "
           "Use scores + chart + quick backtest to decide. (Educational only, not financial advice.)")

# ==============================
# Sidebar — How to use & Glossary
# ==============================
with st.sidebar:
    st.header("🧭 How to use")
    st.markdown("""
1) **Pick tickers** and a **window**.  
2) Tune **Strategy Settings** (RSI / Bollinger %B / Volume / ATR).  
3) See **Top Opportunities** (BuyScore / SellScore).  
4) Open a ticker → **Chart** shows entries/exits and levels.  
5) Quick **backtest** checks the simple rule on this window.
    """)

    st.markdown("---")
    st.subheader("📘 Glossary (simple)")
    st.markdown("""
- **Higher / Lower**: just the **number** going up or down.  
  - Example: **Higher BuyScore** = stronger buy conditions.  
- **Yes / No columns**: a **rule check**.  
  - Example: **Above EMA20 = Yes** means *price is above EMA20* (bullish tilt).  
- **%B (Bollinger)**: position inside the band.  
  - **0.0** = at lower band (cheap side), **1.0** = at upper band (stretched).  
- **Vol Ratio**: today’s volume / 20-day average.  
  - **>1** = more interest than usual (good confirmation).  
- **ATR**: recent daily range size.  
  - Used to set **stop** and **take-profit** distance.
    """)

    st.markdown("---")
    st.subheader("⚙️ Strategy settings")

# ==============================
# Inputs
# ==============================
default_tickers = "HOOD, NVDA, AAPL, MSFT, AMZN, META, AMD, GOOG, TSLA, TSM, JPM, V, SPY, VOO, NOBL, BTOG, LIDR"
tickers_input = st.text_area("Enter stock tickers (comma-separated):", value=default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

period = st.selectbox(
    "Select historical window:",
    options=["1w","2w","3w","1m","2m","3m","6m","1y","2y","3y","5y"],
    index=5 # default 3m
)

top_n = st.slider("Show top opportunities (rows):", 5, 50, 12, 1)

colA, colB, colC = st.columns(3)
with colA:
    rsi_buy_th  = st.slider("RSI oversold ≤", 10, 70, 50, 1)
    rsi_sell_th = st.slider("RSI overbought ≥", 40, 90, 70, 1)
with colB:
    bb_buy_pct  = st.slider("Near lower BB if %B ≤", 0, 50, 20, 1) / 100.0
    bb_sell_pct = st.slider("Near upper BB if %B ≥", 50, 100, 80, 1) / 100.0
with colC:
    vol_ratio_min = st.slider("Min Vol Ratio (x 20-D avg)", 0.3, 3.0, 0.5, 0.1)
    atr_mult      = st.slider("ATR stop multiple", 0.5, 5.0, 2.0, 0.1)

buy_threshold  = st.slider("BuyScore threshold", 0, 100, 60, 1)
sell_threshold = st.slider("SellScore threshold", 0, 100, 60, 1)
rr_target      = st.slider("Take-profit (R multiple)", 8.0, 3.0, 1.0, 0.5)

if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

# ==============================
# Data loader (cached)
# ==============================
@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_data(tickers, window):
    now = datetime.now()
    if window.endswith('w'):
        start = now - pd.DateOffset(weeks=int(window[:-1]))
        df = yf.download(tickers, start=start, end=now, interval="1d", auto_adjust=False, progress=False)
    elif window.endswith('m') and window != "1m":
        start = now - pd.DateOffset(months=int(window[:-1]))
        df = yf.download(tickers, start=start, end=now, interval="1d", auto_adjust=False, progress=False)
    elif window == "1m":
        df = yf.download(tickers, period="1mo", interval="1d", auto_adjust=False, progress=False)
    else:
        df = yf.download(tickers, period=window, interval="1d", auto_adjust=False, progress=False)
    return df

data = fetch_data(tickers, period)
if data is None or data.empty:
    st.error("No data retrieved. Check tickers or connection.")
    st.stop()

# ==============================
# Helpers to extract fields
# ==============================
def get_field(df, field, tlist):
    if isinstance(df.columns, pd.MultiIndex):
        have = [t for t in tlist if t in df[field].columns]
        out = df[field][have].copy()
    else:
        t = tlist[0]
        out = df[[field]].copy()
        out.columns = [t]
    return out

open_df  = get_field(data, "Open", tickers)
high_df  = get_field(data, "High", tickers)
low_df   = get_field(data, "Low", tickers)
close_df = get_field(data, "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close", tickers)
vol_df   = get_field(data, "Volume", tickers)

# drop empty columns (invalid tickers)
valid_cols = [t for t in close_df.columns if close_df[t].dropna().shape[0] > 5]
if not valid_cols:
    st.error("No valid tickers with data.")
    st.stop()
open_df  = open_df[valid_cols]
high_df  = high_df[valid_cols]
low_df   = low_df[valid_cols]
close_df = close_df[valid_cols]
vol_df   = vol_df[valid_cols]

# ==============================
# Indicators
# ==============================
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    m = ema12 - ema26
    s = ema(m, 9)
    h = m - s
    return m, s, h

def bollinger(series, n=20, k=2):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    pb = (series - lower) / (upper - lower)  # %B
    return ma, upper, lower, pb

def atr(high, low, close, n=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def pct_from_52wk_ext(series, window=252):
    roll_high = series.rolling(window).max()
    roll_low  = series.rolling(window).min()
    dist_low  = (series - roll_low) / roll_low * 100
    dist_high = (roll_high - series) / roll_high * 100
    return dist_low, dist_high

def pivots(series):
    mins_mask = (series.shift(1) > series) & (series.shift(-1) > series)
    maxs_mask = (series.shift(1) < series) & (series.shift(-1) < series)
    return mins_mask.fillna(False), maxs_mask.fillna(False)

# ==============================
# Compute per-ticker summary
# ==============================
rows = []
for t in valid_cols:
    p = close_df[t].dropna()
    if p.shape[0] < 30:
        continue
    v = vol_df[t].reindex(p.index).fillna(0)
    h = high_df[t].reindex(p.index)
    l = low_df[t].reindex(p.index)

    rsi14 = rsi(p, 14).iloc[-1]
    macd_line, sig_line, hist = macd(p)
    hist_last     = hist.iloc[-1]
    hist_prev     = hist.iloc[-2] if hist.shape[0] > 1 else np.nan
    macd_cross_up = (hist_prev < 0) and (hist_last > 0)
    macd_cross_dn = (hist_prev > 0) and (hist_last < 0)

    ma20, upBB, loBB, pb = bollinger(p, 20, 2)
    pb_last = float(pb.iloc[-1]) if np.isfinite(pb.iloc[-1]) else np.nan

    ema20 = ema(p, 20).iloc[-1]
    ema50 = ema(p, 50).iloc[-1]
    sma200 = p.rolling(200).mean().iloc[-1]

    atr14 = atr(h, l, p, 14).iloc[-1]
    avg_vol = v.rolling(20).mean().iloc[-1]
    vol_ratio = v.iloc[-1] / avg_vol if avg_vol and avg_vol > 0 else np.nan

    dlow, dhigh = pct_from_52wk_ext(p)
    dist_52w_low  = dlow.iloc[-1] if np.isfinite(dlow.iloc[-1]) else np.nan
    dist_52w_high = dhigh.iloc[-1] if np.isfinite(dhigh.iloc[-1]) else np.nan

    price = p.iloc[-1]
    ret_5d  = (price / p.iloc[-6] - 1) * 100 if p.shape[0] > 6 else np.nan
    ret_21d = (price / p.iloc[-22] - 1) * 100 if p.shape[0] > 22 else np.nan
    ret_ytd = (price / p[p.index.year == p.index[-1].year].iloc[0] - 1) * 100 if (p.index.year == p.index[-1].year).any() else np.nan
    vol_21d = p.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252) * 100  # annualized %

    # --- Scores ---
    score = 0.0
    score += max(0, (rsi_buy_th - rsi14)) / max(1, rsi_buy_th) * 35
    score += max(0.0, (bb_buy_pct - pb_last)) / max(0.001, bb_buy_pct) * 25
    score += 10.0 if macd_cross_up else 0.0
    score += 5.0 if hist_last > 0 else 0.0
    if vol_ratio and vol_ratio > 1:
        score += min(15.0, (vol_ratio - 1) / (2 - 1) * 15.0)
    if dist_52w_low is not np.nan and dist_52w_low < 15:
        score += 8.0
    buy_score = round(min(100.0, score), 1)

    sscore = 0.0
    sscore += max(0, (rsi14 - rsi_sell_th)) / max(1, (100 - rsi_sell_th)) * 35
    sscore += max(0.0, (pb_last - bb_sell_pct)) / max(0.001, (1 - bb_sell_pct)) * 25
    sscore += 10.0 if macd_cross_dn else 0.0
    sscore += 5.0 if hist_last < 0 else 0.0
    if dist_52w_high is not np.nan and dist_52w_high < 10:
        sscore += 10.0
    sell_score = round(min(100.0, sscore), 1)

    if (buy_score >= buy_threshold) and (sell_score < sell_threshold):
        signal = "BUY setup"
    elif (sell_score >= sell_threshold) and (buy_score < buy_threshold):
        signal = "SELL/Trim setup"
    else:
        signal = "WAIT"

    rows.append({
        "Ticker": t,
        "Price": round(price, 2),
        "RSI14": round(float(rsi14), 1),
        "%B": round(float(pb_last), 2) if np.isfinite(pb_last) else np.nan,
        "MACD Hist": round(float(hist_last), 3) if np.isfinite(hist_last) else np.nan,
        "Vol Ratio": round(float(vol_ratio), 2) if np.isfinite(vol_ratio) else np.nan,
        "Ret 5D %": round(ret_5d, 2) if np.isfinite(ret_5d) else np.nan,
        "Ret 21D %": round(ret_21d, 2) if np.isfinite(ret_21d) else np.nan,
        "YTD %": round(ret_ytd, 2) if np.isfinite(ret_ytd) else np.nan,
        "Vol 21D % (ann)": round(vol_21d, 2) if np.isfinite(vol_21d) else np.nan,
        "EMA20": round(float(ema20), 2) if np.isfinite(ema20) else np.nan,
        "EMA50": round(float(ema50), 2) if np.isfinite(ema50) else np.nan,
        "SMA200": round(float(sma200), 2) if np.isfinite(sma200) else np.nan,
        "ATR14": round(float(atr14), 2) if np.isfinite(atr14) else np.nan,
        "Dist 52w Low %": round(dist_52w_low, 2) if np.isfinite(dist_52w_low) else np.nan,
        "Dist 52w High %": round(dist_52w_high, 2) if np.isfinite(dist_52w_high) else np.nan,
        "BuyScore": buy_score,
        "SellScore": sell_score,
        "Signal": signal
    })

summary = pd.DataFrame(rows)
if summary.empty:
    st.error("Not enough data to compute indicators. Try a longer window or different tickers.")
    st.stop()

# ==============================
# Top Opportunities
# ==============================
st.subheader("🏆 Top Opportunities")
left, right = st.columns([2,1])

with left:
    df_buy  = summary[summary["Signal"] == "BUY setup"].sort_values("BuyScore", ascending=False)
    df_sell = summary[summary["Signal"] == "SELL/Trim setup"].sort_values("SellScore", ascending=False)
    df_wait = summary[summary["Signal"] == "WAIT"].sort_values("BuyScore", ascending=False)

    st.markdown("**BUY setups (sorted by BuyScore)**")
    st.dataframe(df_buy.head(top_n), use_container_width=True)
    st.markdown("**SELL/Trim setups (sorted by SellScore)**")
    st.dataframe(df_sell.head(top_n), use_container_width=True)
    st.markdown("**Others (WAIT)**")
    st.dataframe(df_wait.head(max(5, top_n//2)), use_container_width=True)

with right:
    st.markdown("**Sorting / export**")
    sort_by = st.selectbox("Sort whole table by:", ["BuyScore","SellScore","RSI14","%B","Vol Ratio","Ret 21D %","YTD %","Vol 21D % (ann)"], index=0)
    st.download_button("⬇️ Download full table (CSV)", summary.sort_values(sort_by, ascending=False).to_csv(index=False), file_name="summary.csv", mime="text/csv")

# ==============================
# Detailed chart
# ==============================
st.subheader("📊 Detailed Chart & Quick Backtest")
selected = st.selectbox("Choose ticker:", list(summary["Ticker"]), index=0)

p = close_df[selected].dropna()
o = open_df[selected].reindex(p.index)
h = high_df[selected].reindex(p.index)
l = low_df[selected].reindex(p.index)
v = vol_df[selected].reindex(p.index).fillna(0)

ema20_s = ema(p, 20)
ema50_s = ema(p, 50)
sma200_s = p.rolling(200).mean()
ma20_s, upBB_s, loBB_s, pb_s = bollinger(p, 20, 2)
macd_line, sig_line, hist_s = macd(p)
rsi_s = rsi(p, 14)
atr_s = atr(h, l, p, 14)

# local swing points (for visual)
mins_mask, maxs_mask = pivots(p)

# --- Per-bar BuyScore/SellScore for trading logic
def perbar_scores(price, rsi_series, pb_series, hist_series, vol_series):
    volr_series = (vol_series / vol_series.rolling(20).mean()).replace([np.inf, -np.inf], np.nan)
    dlow_s, dhigh_s = pct_from_52wk_ext(price)

    buyS = pd.Series(0.0, index=price.index)
    sellS = pd.Series(0.0, index=price.index)

    for i in range(len(price)):
        # Buy
        bs = 0.0
        rsiv = rsi_series.iloc[i]
        pbv  = pb_series.iloc[i]
        hv   = hist_series.iloc[i]
        vr   = volr_series.iloc[i]
        dl   = dlow_s.iloc[i]
        dh   = dhigh_s.iloc[i]

        if np.isfinite(rsiv):
            bs += max(0, (rsi_buy_th - rsiv)) / max(1, rsi_buy_th) * 35
        if np.isfinite(pbv):
            bs += max(0.0, (bb_buy_pct - pbv)) / max(0.001, bb_buy_pct) * 25
        if np.isfinite(hv):
            bs += 5.0 if hv > 0 else 0.0
        if np.isfinite(vr) and vr > 1:
            bs += min(15.0, (vr - 1) / (2 - 1) * 15.0)
        if np.isfinite(dl) and dl < 15:
            bs += 8.0
        buyS.iloc[i] = min(100.0, bs)

        # Sell
        ss = 0.0
        if np.isfinite(rsiv):
            ss += max(0, (rsiv - rsi_sell_th)) / max(1, (100 - rsi_sell_th)) * 35
        if np.isfinite(pbv):
            ss += max(0.0, (pbv - bb_sell_pct)) / max(0.001, (1 - bb_sell_pct)) * 25
        if np.isfinite(hv):
            ss += 5.0 if hv < 0 else 0.0
        if np.isfinite(dh) and dh < 10:
            ss += 10.0
        sellS.iloc[i] = min(100.0, ss)
    return buyS, sellS

buyS_s, sellS_s = perbar_scores(p, rsi_s, pb_s, hist_s, v)

# --- Trade generator (uses Series; no scalars)
def generate_trades(price, buyS, sellS, buy_th, sell_th, atr_series, atr_m, rr):
    buyS = pd.Series(buyS, index=price.index)
    sellS = pd.Series(sellS, index=price.index)
    atr_series = pd.Series(atr_series, index=price.index)

    entries, exits = [], []
    in_trade = False
    entry_px = None
    stop_px = None
    take_px = None

    for i in range(1, len(price)):
        if not in_trade:
            if (buyS.iloc[i-1] < buy_th) and (buyS.iloc[i] >= buy_th):
                in_trade = True
                entry_px = float(price.iloc[i])
                atr_now  = float(atr_series.iloc[i]) if np.isfinite(atr_series.iloc[i]) else 0.0
                stop_px  = entry_px - atr_m * atr_now
                take_px  = entry_px + rr * atr_m * atr_now
                entries.append((price.index[i], entry_px))
        else:
            px = float(price.iloc[i])
            cond_sell = (sellS.iloc[i-1] < sell_th) and (sellS.iloc[i] >= sell_th)
            cond_stop = (stop_px is not None) and (px <= stop_px)
            cond_take = (take_px is not None) and (px >= take_px)
            if cond_sell or cond_stop or cond_take:
                exits.append((price.index[i], px))
                in_trade = False
                entry_px = stop_px = take_px = None

    if in_trade and entry_px is not None:
        exits.append((price.index[-1], float(price.iloc[-1])))
    return entries, exits

entries, exits = generate_trades(p, buyS_s, sellS_s, buy_threshold, sell_threshold, atr_s, atr_mult, rr_target)

# --- Backtest
def quick_backtest(price, entries, exits):
    n = min(len(entries), len(exits))
    pairs = list(zip(entries[:n], exits[:n]))
    trades = []
    for (dt_e, px_e), (dt_x, px_x) in pairs:
        r = (px_x / px_e - 1.0)
        trades.append({"Entry": dt_e, "EntryPx": px_e, "Exit": dt_x, "ExitPx": px_x, "Return": r})
    if not trades:
        return pd.DataFrame([]), 0.0, 0.0, 0.0, 0.0, 0.0

    trdf = pd.DataFrame(trades).sort_values("Entry")
    total_ret = (trdf["Return"] + 1).prod() - 1
    winrate = (trdf["Return"] > 0).mean() * 100
    avg_gain = trdf.loc[trdf["Return"]>0, "Return"].mean()*100 if (trdf["Return"]>0).any() else 0.0
    avg_loss = trdf.loc[trdf["Return"]<0, "Return"].mean()*100 if (trdf["Return"]<0).any() else 0.0

    eq = (trdf["Return"] + 1).cumprod()
    peak = eq.cummax()
    max_dd = ((eq/peak) - 1).min()*100 if len(eq) else 0.0

    return trdf, total_ret*100, winrate, avg_gain, avg_loss, max_dd

trades_df, tot_ret, winrate, avg_gain, avg_loss, max_dd = quick_backtest(p, entries, exits)

# --- Plotly figure (4 rows)
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.45, 0.15, 0.2, 0.2],
    subplot_titles=(
        f"{selected} Candles + MAs + Bollinger",
        "Volume",
        "MACD",
        "RSI"
    )
)

# Row 1: Candlestick + MAs + BB + swing points + signal markers
fig.add_trace(go.Candlestick(x=p.index, open=o, high=h, low=l, close=p, name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=p.index, y=ema20_s, name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=p.index, y=ema50_s, name="EMA50"), row=1, col=1)
fig.add_trace(go.Scatter(x=p.index, y=sma200_s, name="SMA200", line=dict(dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=p.index, y=upBB_s, name="Upper BB", line=dict(dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=p.index, y=loBB_s, name="Lower BB", line=dict(dash="dot")), row=1, col=1)

mins_mask, maxs_mask = pivots(p)
fig.add_trace(go.Scatter(x=p.index[mins_mask], y=p[mins_mask], mode="markers", name="Swing Low",
                         marker=dict(symbol="triangle-up", size=9)), row=1, col=1)
fig.add_trace(go.Scatter(x=p.index[maxs_mask], y=p[maxs_mask], mode="markers", name="Swing High",
                         marker=dict(symbol="triangle-down", size=9)), row=1, col=1)

if entries:
    fig.add_trace(go.Scatter(x=[d for d,_ in entries], y=[px for _,px in entries],
                             mode="markers", name="Entry", marker=dict(symbol="arrow-up", size=12)), row=1, col=1)
if exits:
    fig.add_trace(go.Scatter(x=[d for d,_ in exits], y=[px for _,px in exits],
                             mode="markers", name="Exit", marker=dict(symbol="arrow-down", size=12)), row=1, col=1)

# Row 2: Volume
fig.add_trace(go.Bar(x=p.index, y=v, name="Volume"), row=2, col=1)

# Row 3: MACD
fig.add_trace(go.Scatter(x=p.index, y=macd_line, name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=p.index, y=sig_line,  name="Signal"), row=3, col=1)
fig.add_trace(go.Bar(x=p.index, y=hist_s, name="Hist", opacity=0.6), row=3, col=1)

# Row 4: RSI
fig.add_trace(go.Scatter(x=p.index, y=rsi_s, name="RSI14"), row=4, col=1)
fig.add_trace(go.Scatter(x=p.index, y=[70]*len(p), name="70", line=dict(dash="dash")), row=4, col=1)
fig.add_trace(go.Scatter(x=p.index, y=[30]*len(p), name="30", line=dict(dash="dash")), row=4, col=1)

fig.update_layout(height=950, showlegend=True, xaxis_rangeslider_visible=False,
                  title_text=f"{selected} — strategy signals ({period})")
st.plotly_chart(fig, use_container_width=True)

# ==============================
# Backtest summary
# ==============================
st.markdown("### 🧪 Quick Backtest (very simple, long-only)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Return", f"{tot_ret:.2f}%")
c2.metric("Win Rate", f"{winrate:.1f}%")
c3.metric("Avg Win", f"{avg_gain:.2f}%")
c4.metric("Avg Loss", f"{avg_loss:.2f}%")
c5.metric("Max Drawdown", f"{max_dd:.2f}%")
st.caption(
    "Rules: Enter when **BuyScore** crosses above threshold; exit on **SellScore** cross, or "
    f"stop at {atr_mult}×ATR, or take-profit at {rr_target}×ATR. Only for this window."
)

if not trades_df.empty:
    st.dataframe(trades_df, use_container_width=True)
else:
    st.info("No trades triggered by the current thresholds in this window. Try lowering BuyScore or increasing window.")

# ==============================
# Extra tips
# ==============================
with st.expander("💡 Reading the signals (quick tips)"):
    st.markdown("""
- **BUY setup** is stronger when: BuyScore high **and rising**, **MACD histogram turns positive**, **%B low (~0–0.2)**, **Vol Ratio > 1**.  
- **SELL/Trim** when: SellScore high, **%B near 1.0**, **RSI > 70**, **MACD histogram falling/negative**.  
- **Wait** when signals conflict or volume is weak.  
- Try combining this with **fundamentals** and **news** before making a trade.
    """)