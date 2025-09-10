# -*- coding: utf-8 -*-
##############################################################################################
# Stock Decision Dashboard — buy low / sell high helper (not financial advice)
# Run: streamlit run app.py
##############################################################################################

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ==============================
# Page / Mobile UX (single call)
# ==============================
st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")

# ---- Sidebar (quick how-to + UX toggles)
with st.sidebar:
    st.header("🧭 How to use")
    st.markdown(
        "1) Pick **tickers** and **window**.\n"
        "2) Tune **Strategy** (RSI main; others optional in Advanced).\n"
        "3) See **Top Opportunities**.\n"
        "4) Open a ticker → **Chart** + quick backtest.\n\n"
        "_Educational only — not financial advice._"
    )
    st.markdown("---")
    compact = st.toggle("📱 Compact mobile layout", value=True)
    simple_view_default = st.toggle("🧼 Simple chart view by default", value=True)
    # Use this when you scan pre-market / quiet sessions
    lenient_mode = st.toggle("🌙 Lenient / pre-market mode", value=True)

# ---- Global CSS for mobile readability
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 0.5rem; padding-bottom: 2rem;}
@media (max-width: 768px) {
  .block-container {padding: 0.5rem 0.7rem;}
  .stDataFrame {font-size: 1.0rem;}
  .stButton>button {width: 100%; min-height: 48px; font-size: 1.05rem;}
  .stTextInput>div>div>input, textarea, select {font-size: 1.05rem !important;}
}
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
if compact:
    st.markdown("<style>.element-container{margin-bottom:0.6rem;}</style>", unsafe_allow_html=True)

# ==============================
# Header
# ==============================
st.title("📈 Stock Decision Dashboard")
st.caption("Find **buy** near local lows and **sell/trim** near local highs. Use scores + charts + quick backtest. (Not financial advice)")

# ==============================
# Inputs
# ==============================
default_tickers = (
    "HOOD, NVDA, AAPL, MSFT, AMZN, META, AMD, GOOG, TSLA, TSM, JPM, V, SPY, VOO, NOBL, INTC, "
    "PLTR, SMCI, APP, SE, SHOP, NET, CEG, VST, NRG, NEE, AVGO, LLY, CRM, INTU, CTAS, HEI, EQT, ROAD, MP"
)
tickers_input = st.text_area("Enter stock tickers (comma-separated):", value=default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

period = st.selectbox(
    "Select historical window:",
    options=["1w","2w","3w","1m","2m","3m","6m","1y","2y","3y","5y"],
    index=5  # default 3m
)

top_n = st.slider("Show top opportunities (rows):", 5, 50, 12, 1)

# ---- Key sliders (kept visible)
colA1, colA2 = st.columns(2)
with colA1:
    rsi_buy_th  = st.slider("RSI oversold ≤", 10, 70, 55, 1)   # softer default
with colA2:
    rsi_sell_th = st.slider("RSI overbought ≥", 40, 90, 65, 1) # softer default

# ---- Less critical → Advanced (hidden by default, has good defaults)
with st.expander("⚙️ Advanced strategy settings (optional)", expanded=False):
    colB, colC = st.columns(2)
    with colB:
        bb_buy_pct  = st.slider("Near lower BB if %B ≤", 0, 60, 35, 1) / 100.0
        bb_sell_pct = st.slider("Near upper BB if %B ≥", 50, 100, 75, 1) / 100.0
        vol_ratio_min = st.slider("Min Vol Ratio (x 20D avg)", 0.1, 3.0, 0.4, 0.1)
    with colC:
        atr_mult      = st.slider("ATR stop multiple", 0.5, 5.0, 1.5, 0.1)
        buy_threshold  = st.slider("BuyScore threshold", 0, 100, 50, 1)
        sell_threshold = st.slider("SellScore threshold", 0, 100, 55, 1)
        rr_target      = st.slider("Take-profit (R multiple)", 0.5, 5.0, 1.5, 0.5)
        use_volume_gate = st.checkbox("Require min Vol Ratio (gate entries)", value=False)

# sensible defaults if user keeps Advanced closed (values above already set)

# ---- Auto relax a bit more in quiet hours
if lenient_mode:
    rsi_buy_th  = min(70, rsi_buy_th + 5)       # easier to be 'oversold'
    rsi_sell_th = max(40, rsi_sell_th - 5)      # easier to be 'overbought'
    bb_buy_pct  = min(0.6, bb_buy_pct + 0.10)   # allow higher %B on buys
    bb_sell_pct = max(0.5, bb_sell_pct - 0.05)  # allow lower %B on sells
    buy_threshold  = max(20, buy_threshold - 10)
    sell_threshold = max(20, sell_threshold - 5)
    vol_ratio_min  = max(0.1, vol_ratio_min - 0.1)

if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

# ==============================
# 📘 Guidelines & Parameter Cheatsheet (updated)
# ==============================
with st.expander("📘 Guidelines & Parameter Cheatsheet (what everything means & how it’s scored)", expanded=False):
    st.markdown(f"""
**A. Core indicators**
- **RSI14**: 0–100. Lower = more oversold.  
  • **Buy idea**: RSI ≤ **{rsi_buy_th}** (oversold).  
  • **Sell idea**: RSI ≥ **{rsi_sell_th}** (overbought).

- **Bollinger %B**: position inside band (0≈lower, 1≈upper).  
  • **Buy** if %B ≤ **{bb_buy_pct:.2f}**; **Sell** if %B ≥ **{bb_sell_pct:.2f}**.

- **MACD**: momentum flips via histogram / cross.  
  • Bullish when hist ↗ through 0; Bearish when ↘ through 0.

- **Golden Cross / Death Cross (NEW)**:  
  • **Golden Cross** when **SMA50 > SMA200** (bullish bias).  
  • **Death Cross** when **SMA50 < SMA200** (bearish bias).  
  • A **recent cross** adds conviction to the signal.

- **Stochastic (14,3,3) (NEW)**:  
  • **%K** and **%D**. **<20** = oversold (buy), **>80** = overbought (sell).  
  • Often faster than RSI.

- **ADX(14) (NEW)**: trend **strength** (not direction).  
  • **>25** = strong trend; **<20** = choppy.  
  • Use with MACD: strong + bullish = higher confidence.

**B. Volume & Risk**
- **Vol Ratio**: today’s vol ÷ 20D avg.  
  • >1 = active interest; <1 = quiet. Optionally gate entries (min **{vol_ratio_min:.2f}**).
- **ATR14 × {atr_mult:.2f}**: sets stop; **TP = ATR × {atr_mult:.2f} × {rr_target:.2f}**.

**C. Scores & Labels**
- **BuyScore ≥ {buy_threshold}** and **SellScore < {sell_threshold}** → **BUY setup**  
- **SellScore ≥ {sell_threshold}** and **BuyScore < {buy_threshold}** → **SELL/Trim**  
- Else → WAIT.

**D. Quick recipes**
- *Mean-reversion buy*: RSI ≤ 45–55, %B ≤ 0.20–0.35, BuyScore ≥ 45–60, Vol Ratio > 0.8.  
- *Momentum trim*: RSI ≥ 65–70, %B ≥ 0.75–1.0, SellScore ≥ 55–65, near 52w high, ADX > 25.
""")

# ==============================
# Data loader (cached)
# ==============================
@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_data(tickers, window):
    """Download OHLCV for given tickers and window."""
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
    """Handle yfinance's MultiIndex vs single-index columns."""
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

# ---- Drop tickers with too few rows
valid_cols = [t for t in close_df.columns if close_df[t].dropna().shape[0] > 30]
if not valid_cols:
    st.error("No valid tickers with enough data.")
    st.stop()
open_df  = open_df[valid_cols]
high_df  = high_df[valid_cols]
low_df   = low_df[valid_cols]
close_df = close_df[valid_cols]
vol_df   = vol_df[valid_cols]

# ==============================
# Indicators (NEW: Stochastic & ADX; Golden Cross)
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
    width = (upper - lower).replace(0, np.nan)
    pb = (series - lower) / width   # %B in [0,1] ideally
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
    dist_low  = (series - roll_low) / roll_low.replace(0, np.nan) * 100
    dist_high = (roll_high - series) / roll_high.replace(0, np.nan) * 100
    return dist_low, dist_high

def pivots(series):
    """Very quick local swing points: prev>cur<next (lows) and prev<cur>next (highs)."""
    mins_mask = (series.shift(1) > series) & (series.shift(-1) > series)
    maxs_mask = (series.shift(1) < series) & (series.shift(-1) < series)
    return mins_mask.fillna(False), maxs_mask.fillna(False)

def heikin_ashi(o,h,l,c):
    """Optional visual smoothing for noisy candles (chart only)."""
    ha_close = (o + h + l + c) / 4
    ha_open = pd.Series(index=o.index, dtype=float)
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(o)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close

# --- NEW: Stochastic (14,3,3)
def stochastic(high, low, close, n=14, k=3, d=3):
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    raw_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    K = raw_k.rolling(k).mean()
    D = K.rolling(d).mean()
    return K, D

# --- NEW: ADX(14)
def adx(high, low, close, n=14):
    up = high.diff()
    dn = low.diff().abs()
    plus_dm  = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (low.diff() < 0)).astype(float) * dn

    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_ = tr.rolling(n).mean()
    plus_di  = 100 * (plus_dm.rolling(n).mean() / atr_)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr_)
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx_val = dx.rolling(n).mean()
    return adx_val, plus_di, minus_di

# ==============================
# Per-ticker summary (scores + key fields)
# ==============================
rows = []
for t in valid_cols:
    p = close_df[t].dropna()
    v = vol_df[t].reindex(p.index).fillna(0)
    h = high_df[t].reindex(p.index)
    l = low_df[t].reindex(p.index)

    # Indicators
    rsi14 = rsi(p, 14).iloc[-1]
    macd_line, sig_line, hist = macd(p)
    hist_last = hist.iloc[-1]
    hist_prev = hist.iloc[-2] if hist.shape[0] > 1 else np.nan
    macd_cross_up = (pd.notna(hist_prev) and hist_prev < 0) and (pd.notna(hist_last) and hist_last > 0)
    macd_cross_dn = (pd.notna(hist_prev) and hist_prev > 0) and (pd.notna(hist_last) and hist_last < 0)

    ma20, upBB, loBB, pb = bollinger(p, 20, 2)
    pb_last = float(pb.iloc[-1]) if pd.notna(pb.iloc[-1]) and np.isfinite(pb.iloc[-1]) else np.nan

    ema20 = ema(p, 20).iloc[-1]
    ema50 = ema(p, 50).iloc[-1]
    sma50_series = p.rolling(50).mean()
    sma200_series = p.rolling(200).mean()
    sma50 = sma50_series.iloc[-1]
    sma200 = sma200_series.iloc[-1]

    # Golden/Death cross
    golden_now = (sma50 > sma200) if pd.notna(sma50) and pd.notna(sma200) else False
    spread = sma50_series - sma200_series
    cross_up_recent = (spread.tail(6).iloc[0] <= 0) and (spread.tail(6).iloc[-1] > 0) if spread.dropna().shape[0] > 6 else False
    cross_dn_recent = (spread.tail(6).iloc[0] >= 0) and (spread.tail(6).iloc[-1] < 0) if spread.dropna().shape[0] > 6 else False

    atr14 = atr(h, l, p, 14).iloc[-1]
    avg_vol = v.rolling(20).mean().iloc[-1]
    vol_ratio = (v.iloc[-1] / avg_vol) if (pd.notna(avg_vol) and avg_vol > 0) else np.nan

    dlow, dhigh = pct_from_52wk_ext(p)
    dist_52w_low  = dlow.iloc[-1] if pd.notna(dlow.iloc[-1]) else np.nan
    dist_52w_high = dhigh.iloc[-1] if pd.notna(dhigh.iloc[-1]) else np.nan

    # NEW: Stochastic & ADX
    stochK, stochD = stochastic(h, l, p, 14, 3, 3)
    stochK_last = stochK.iloc[-1]
    stochD_last = stochD.iloc[-1]
    adx14, plusDI, minusDI = adx(h, l, p, 14)
    adx_last = adx14.iloc[-1]

    price = p.iloc[-1]
    ret_21d = (price / p.iloc[-22] - 1) * 100 if p.shape[0] > 22 else np.nan
    try:
        ymask = (p.index.year == p.index[-1].year)
        ret_ytd = (price / p[ymask].iloc[0] - 1) * 100 if ymask.any() else np.nan
    except Exception:
        ret_ytd = np.nan
    vol_21d = p.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252) * 100  # annualized %

    # ----- Buy/Sell Scores (add light contributions from Stoch & ADX)
    score = 0.0
    if pd.notna(rsi14):
        score += max(0, (rsi_buy_th - rsi14)) / max(1, rsi_buy_th) * 35
    if pd.notna(pb_last):
        score += max(0.0, (bb_buy_pct - pb_last)) / max(0.001, bb_buy_pct) * 25
    if macd_cross_up:
        score += 10.0
    if pd.notna(hist_last) and (hist_last > 0):
        score += 5.0
    if pd.notna(stochK_last) and (stochK_last < 20):
        score += 5.0
    if pd.notna(adx_last) and (adx_last > 25) and (pd.notna(hist_last) and hist_last > 0):
        score += 3.0
    if pd.notna(vol_ratio):
        if vol_ratio > 1:
            score += min(15.0, (vol_ratio - 1) / (2 - 1) * 15.0)
        elif use_volume_gate and (vol_ratio < vol_ratio_min):
            score -= 8.0
    if pd.notna(dist_52w_low) and (dist_52w_low < 20):
        score += 8.0
    buy_score = round(min(100.0, max(0.0, score)), 1)

    sscore = 0.0
    if pd.notna(rsi14):
        sscore += max(0, (rsi14 - rsi_sell_th)) / max(1, (100 - rsi_sell_th)) * 35
    if pd.notna(pb_last):
        sscore += max(0.0, (pb_last - bb_sell_pct)) / max(0.001, (1 - bb_sell_pct)) * 25
    if macd_cross_dn:
        sscore += 10.0
    if pd.notna(hist_last) and (hist_last < 0):
        sscore += 5.0
    if pd.notna(stochK_last) and (stochK_last > 80):
        sscore += 5.0
    if pd.notna(adx_last) and (adx_last > 25) and (pd.notna(hist_last) and hist_last < 0):
        sscore += 3.0
    if pd.notna(dist_52w_high) and (dist_52w_high < 12):
        sscore += 10.0
    sell_score = round(min(100.0, max(0.0, sscore)), 1)

    # ----- Signal label
    bt, stt = buy_threshold, sell_threshold
    if (buy_score >= bt) and (sell_score < stt):
        signal = "BUY setup"
    elif (sell_score >= stt) and (buy_score < bt):
        signal = "SELL/Trim setup"
    else:
        signal = "WAIT"

    rows.append({
        # Decision-first ordering (important columns first)
        "Ticker": t,
        "Signal": signal,
        "BuyScore": buy_score,
        "SellScore": sell_score,
        "Price": round(float(price), 2),
        "RSI14": round(float(rsi14), 1) if pd.notna(rsi14) else np.nan,
        "%B": round(float(pb_last), 2) if pd.notna(pb_last) else np.nan,
        "Stoch %K": round(float(stochK_last), 1) if pd.notna(stochK_last) else np.nan,
        "Stoch %D": round(float(stochD_last), 1) if pd.notna(stochD_last) else np.nan,
        "ADX14": round(float(adx_last), 1) if pd.notna(adx_last) else np.nan,
        "Golden Cross": "Yes" if golden_now else "No",
        "GCross Recent": "Yes" if cross_up_recent else ("Death Recent" if cross_dn_recent else "No"),
        "Ret 21D %": round(ret_21d, 2) if pd.notna(ret_21d) else np.nan,
        "YTD %": round(ret_ytd, 2) if pd.notna(ret_ytd) else np.nan,
        "EMA20": round(float(ema20), 2) if pd.notna(ema20) else np.nan,
        "SMA200": round(float(sma200), 2) if pd.notna(sma200) else np.nan,
    })

summary = pd.DataFrame(rows)
if summary.empty:
    st.error("Not enough data to compute indicators. Try a longer window or different tickers.")
    st.stop()

# ---- Reorder to decision-first columns for display
priority_cols = [
    "Ticker","Signal","BuyScore","SellScore","Price",
    "RSI14","%B","Stoch %K","Stoch %D","ADX14","Golden Cross","GCross Recent",
    "Ret 21D %","YTD %","EMA20","SMA200"
]
summary = summary[priority_cols + [c for c in summary.columns if c not in priority_cols]]

# ==============================
# Top Opportunities
# ==============================
st.subheader("🏆 Top Opportunities")

left, right = st.columns([2,1])
with right:
    sort_by = st.selectbox("Sort whole table by:",
                           ["BuyScore","SellScore","RSI14","%B","Stoch %K","ADX14","Ret 21D %","YTD %"],
                           index=0)
    filt = st.radio("Quick filter:", ["All","Only BUY","Only SELL"], index=0, horizontal=True)

with left:
    df_sorted = summary.sort_values(sort_by, ascending=False)
    if filt == "Only BUY":
        df_sorted = df_sorted[df_sorted["Signal"]=="BUY setup"]
    elif filt == "Only SELL":
        df_sorted = df_sorted[df_sorted["Signal"]=="SELL/Trim setup"]
    st.dataframe(df_sorted.head(top_n), use_container_width=True, hide_index=True)

# ==============================
# Detailed chart & quick backtest
# ==============================
st.subheader("📊 Detailed Chart & Quick Backtest")
selected = st.selectbox("Choose ticker:", list(summary["Ticker"]), index=0)

# ---- Slice selected series
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
stochK_s, stochD_s = stochastic(h, l, p, 14, 3, 3)
adx_s, plusDI_s, minusDI_s = adx(h, l, p, 14)
atr_s = atr(h, l, p, 14)
mins_mask, maxs_mask = pivots(p)

# ---- Per-bar Buy/Sell Scores (same as before, with volume/lenient applied)
def perbar_scores(price, rsi_series, pb_series, hist_series, vol_series):
    volr_series = (vol_series / vol_series.rolling(20).mean()).replace([np.inf, -np.inf], np.nan)
    dlow_s, dhigh_s = pct_from_52wk_ext(price)
    buyS = pd.Series(0.0, index=price.index, dtype=float)
    sellS = pd.Series(0.0, index=price.index, dtype=float)

    for i in range(len(price)):
        rsiv = rsi_series.iloc[i]
        pbv  = pb_series.iloc[i]
        hv   = hist_series.iloc[i]
        vr   = volr_series.iloc[i]
        dl   = dlow_s.iloc[i]
        dh   = dhigh_s.iloc[i]

        bs = 0.0
        if pd.notna(rsiv):
            bs += max(0, (rsi_buy_th - rsiv)) / max(1, rsi_buy_th) * 35
        if pd.notna(pbv):
            bs += max(0.0, (bb_buy_pct - pbv)) / max(0.001, bb_buy_pct) * 25
        if pd.notna(hv) and (hv > 0): bs += 5.0
        if pd.notna(vr):
            if vr > 1: bs += min(15.0, (vr - 1) / (2 - 1) * 15.0)
            elif use_volume_gate and (vr < vol_ratio_min): bs -= 8.0
        if pd.notna(dl) and (dl < 20): bs += 8.0
        buyS.iloc[i] = min(100.0, max(0.0, bs))

        ss = 0.0
        if pd.notna(rsiv):
            ss += max(0, (rsiv - rsi_sell_th)) / max(1, (100 - rsi_sell_th)) * 35
        if pd.notna(pbv):
            ss += max(0.0, (pbv - bb_sell_pct)) / max(0.001, (1 - bb_sell_pct)) * 25
        if pd.notna(hv) and (hv < 0): ss += 5.0
        if pd.notna(dh) and (dh < 12): ss += 10.0
        sellS.iloc[i] = min(100.0, max(0.0, ss))
    return buyS, sellS

buyS_s, sellS_s = perbar_scores(p, rsi_s, pb_s, hist_s, v)

# ---- Threshold mode (manual vs adaptive)
threshold_mode = st.radio(
    "Threshold mode (chart/backtest):",
    ["Manual (sliders above)", "Adaptive from recent scores (80th percentile)"],
    index=0, horizontal=False
)

if threshold_mode.startswith("Adaptive"):
    recent = max(40, int(len(buyS_s) * 0.5))  # focus on recent regime
    bq = float(buyS_s.tail(recent).quantile(0.80))
    sq = float(sellS_s.tail(recent).quantile(0.80))
    buy_th_use = max(20, round(bq, 1))
    sell_th_use = max(20, round(sq, 1))
    st.caption(f"Adaptive BuyScore≈{buy_th_use}, SellScore≈{sell_th_use} (80th pct of last {recent} bars)")
else:
    buy_th_use = buy_threshold
    sell_th_use = sell_threshold

# ---- Trade generator (simple cross + ATR stop/take)
def generate_trades(price, buyS, sellS, buy_th, sell_th, atr_series, atr_m, rr):
    entries, exits = [], []
    in_trade = False
    entry_px = stop_px = take_px = None

    for i in range(1, len(price)):
        if not in_trade:
            if (buyS.iloc[i-1] < buy_th) and (buyS.iloc[i] >= buy_th):
                in_trade = True
                entry_px = float(price.iloc[i])
                atr_now  = float(atr_series.iloc[i]) if pd.notna(atr_series.iloc[i]) else 0.0
                stop_px  = entry_px - atr_m * atr_now if atr_now > 0 else None
                take_px  = entry_px + rr * atr_m * atr_now if atr_now > 0 else None
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

entries, exits = generate_trades(p, buyS_s, sellS_s, buy_th_use, sell_th_use, atr_s, atr_mult, rr_target)

# ---- Backtest stats
def quick_backtest(price, entries, exits):
    n = min(len(entries), len(exits))
    if n == 0:
        return pd.DataFrame([]), 0.0, 0.0, 0.0, 0.0, 0.0
    pairs = list(zip(entries[:n], exits[:n]))
    trades = []
    for (dt_e, px_e), (dt_x, px_x) in pairs:
        r = (px_x / px_e - 1.0)
        trades.append({"Entry": dt_e, "EntryPx": px_e, "Exit": dt_x, "ExitPx": px_x, "Return": r})
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

# ==============================
# Chart controls (reduce clutter on phone)
# ==============================
st.markdown("#### Chart options")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1:
    simple_view = st.checkbox("Simple view", value=simple_view_default)
with c2:
    show_bb = st.checkbox("Bollinger", value=not simple_view)
with c3:
    show_mas = st.checkbox("EMA50 / SMA200", value=not simple_view)
with c4:
    show_macd = st.checkbox("MACD panel", value=not simple_view)
with c5:
    show_stoch = st.checkbox("Stochastic panel", value=False)
with c6:
    show_adx = st.checkbox("ADX panel", value=False)
with c7:
    show_vol = st.checkbox("Volume panel", value=not simple_view)

# ---- Decide subplot layout dynamically
rows = 1 + int(show_vol) + int(show_macd) + int(show_stoch) + int(show_adx) + 1  # price + optional panels + rsi
if simple_view:
    base = 0.55
else:
    base = 0.45
row_heights = [base]
for flag in [show_vol, show_macd, show_stoch, show_adx]:
    if flag: row_heights.append(0.14)
row_heights.append(0.15)  # RSI

titles = ["Price"]
if show_vol:   titles.append("Volume")
if show_macd:  titles.append("MACD")
if show_stoch: titles.append("Stochastic")
if show_adx:   titles.append("ADX")
titles.append("RSI")

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=row_heights, subplot_titles=titles
)

# ---- Row indices
row_idx = 1

# ---- Price row (candles + key overlays + markers)
fig.add_trace(go.Candlestick(x=p.index, open=o, high=h, low=l, close=p, name="Price"),
              row=row_idx, col=1)

fig.add_trace(go.Scatter(x=p.index, y=ema20_s, name="EMA20", line=dict(width=1.6)), row=row_idx, col=1)
if show_mas:
    fig.add_trace(go.Scatter(x=p.index, y=ema50_s, name="EMA50", line=dict(width=1.2)), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=sma200_s, name="SMA200", line=dict(dash="dash", width=1.2)), row=row_idx, col=1)
if show_bb:
    fig.add_trace(go.Scatter(x=p.index, y=upBB_s, name="Upper BB", line=dict(dash="dot", width=1)), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=loBB_s, name="Lower BB", line=dict(dash="dot", width=1)), row=row_idx, col=1)

# ---- Volume
if show_vol:
    row_idx += 1
    fig.add_trace(go.Bar(x=p.index, y=v, name="Volume", opacity=0.8), row=row_idx, col=1)

# ---- MACD
if show_macd:
    row_idx += 1
    fig.add_trace(go.Scatter(x=p.index, y=macd_line, name="MACD"), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=sig_line,  name="Signal"), row=row_idx, col=1)
    fig.add_trace(go.Bar(x=p.index, y=hist_s, name="Hist", opacity=0.6), row=row_idx, col=1)

# ---- Stochastic
if show_stoch:
    row_idx += 1
    fig.add_trace(go.Scatter(x=p.index, y=stochK_s, name="%K"), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=stochD_s, name="%D"), row=row_idx, col=1)
    fig.add_hline(y=80, line_dash="dash", row=row_idx, col=1)
    fig.add_hline(y=20, line_dash="dash", row=row_idx, col=1)

# ---- ADX
if show_adx:
    row_idx += 1
    fig.add_trace(go.Scatter(x=p.index, y=adx_s, name="ADX(14)"), row=row_idx, col=1)
    fig.add_hline(y=25, line_dash="dash", row=row_idx, col=1)

# ---- RSI (always shown)
row_idx += 1
fig.add_trace(go.Scatter(x=p.index, y=rsi_s, name="RSI14", line=dict(width=1.6)), row=row_idx, col=1)
fig.add_trace(go.Scatter(x=p.index, y=[70]*len(p), name="70", line=dict(dash="dash", width=1)), row=row_idx, col=1)
fig.add_trace(go.Scatter(x=p.index, y=[30]*len(p), name="30", line=dict(dash="dash", width=1)), row=row_idx, col=1)

# ---- Layout for readability on phone
fig.update_layout(
    height=820 if simple_view else 1020,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    title_text=f"{selected} — signals ({period})",
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(size=14 if compact else 13),
    dragmode="pan"
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

# ==============================
# Backtest summary (simple, long-only)
# ==============================
st.markdown("### 🧪 Quick Backtest")
c1, c2, c3, c4, c5 = st.columns(5)
trades_df, tot_ret, winrate, avg_gain, avg_loss, max_dd = quick_backtest(p, entries, exits)
c1.metric("Total Return", f"{tot_ret:.2f}%")
c2.metric("Win Rate", f"{winrate:.1f}%")
c3.metric("Avg Win", f"{avg_gain:.2f}%")
c4.metric("Avg Loss", f"{avg_loss:.2f}%")
c5.metric("Max Drawdown", f"{max_dd:.2f}%")
st.caption(
    f"Rules: Enter when BuyScore crosses **above** {buy_th_use}; exit on SellScore **above** {sell_th_use}, "
    f"or stop at {atr_mult}×ATR, or take-profit at {rr_target}×ATR. Window-limited backtest."
)

if not trades_df.empty:
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
else:
    st.info("No trades with current thresholds. Try **Lenient mode** or **Adaptive thresholds**.")

# ==============================
# Tips (short)
# ==============================
with st.expander("💡 Quick tips"):
    st.markdown(
        "- **BUY**: BuyScore rising, MACD hist flips positive, %B ~ 0–0.35, RSI below threshold, Stoch %K < 20, ADX>25 helpful, Vol Ratio > 1 if possible.\n"
        "- **SELL/Trim**: SellScore high, %B ~ 0.75–1.0, RSI > 65–70, Stoch %K > 80, MACD hist falling/negative, near 52w high.\n"
        "- **Golden Cross** adds long bias; **Death Cross** adds caution.\n"
        "- Quiet sessions → **Lenient mode** or **Adaptive thresholds**.\n"
        "- Always cross-check with broader context (earnings, news)."
    )
