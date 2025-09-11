# -*- coding: utf-8 -*-
##############################################################################################
# Stock Decision Dashboard — buy low / sell high helper (not financial advice)
#
# NEW in this version
#   • News sentiment (Yahoo Finance + Google News RSS fallback) per ticker
#   • Macro regime: VIX percentile, SPY trend, BTC/ETH 24h (risk-on proxy)
#   • Policy-risk keywords (e.g., tariff / Trump / SEC / lawsuit) with penalty
#   • Real-time Decision Panel: Probability the next move is UP (gauge + reasons)
#   • "🔄 Refresh" button clears cache and refetches data/news immediately
#   • Minor fixes, better error handling, and mobile-friendly tweaks
#
# Run: streamlit run app.py
##############################################################################################

import math
import time
import json
import requests
from urllib.parse import quote_plus
from datetime import datetime, timedelta, timezone

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

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
        "4) Open a ticker → **Chart** + backtest + **Decision Panel**.\n\n"
        "_Educational only — not financial advice._"
    )
    st.markdown("---")
    compact = st.toggle("📱 Compact mobile layout", value=True)
    simple_view_default = st.toggle("🧼 Simple chart view by default", value=True)
    lenient_mode = st.toggle("🌙 Lenient / pre-market mode", value=True)

    st.markdown("---")
    if st.button("🔄 Refresh data & news (force)"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        # Trigger a rerun
        st.session_state["_refresh_ts"] = time.time()
        st.rerun()

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
st.title("📈 Stock Decision Dashboard — with News & Probability")
st.caption("Find **buy** near local lows and **sell/trim** near local highs. Scores + news + quick backtest + probability. (Not financial advice)")

# ==============================
# Inputs
# ==============================
default_tickers = (
    "HOOD, NVDA, AAPL, MSFT, AMZN, META, AMD, GOOG, TSLA, TSM, JPM, V, SPY, VOO, NOBL, INTC, "
    "PLTR, SMCI, APP, SE, SHOP, NET, CEG, VST, NRG, NEE, AVGO, LLY, CRM, INTU, CTAS, HEI, EQT, ROAD, MP"
)

tickers_input = st.text_area("Enter stock tickers (comma-separated):", value=default_tickers)
# sanitize and unique
_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
# keep order but unique
seen = set()
tickers = []
for t in _tickers:
    if t not in seen:
        tickers.append(t)
        seen.add(t)

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
# Utilities
# ==============================
@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_prices(tickers, window):
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

@st.cache_data(show_spinner=False, ttl=60*5)
def fetch_macro_series():
    """Get macro proxies: VIX, SPY, BTC, ETH (last values and small window)."""
    macro = {}
    try:
        vix = yf.download("^VIX", period="6mo", interval="1d", auto_adjust=False, progress=False)
        macro["vix"] = vix
    except Exception:
        macro["vix"] = pd.DataFrame()
    try:
        spy = yf.download("SPY", period="3mo", interval="1d", auto_adjust=False, progress=False)
        macro["spy"] = spy
    except Exception:
        macro["spy"] = pd.DataFrame()
    try:
        btc = yf.download("BTC-USD", period="7d", interval="1h", auto_adjust=False, progress=False)
        eth = yf.download("ETH-USD", period="7d", interval="1h", auto_adjust=False, progress=False)
        macro["btc"] = btc
        macro["eth"] = eth
    except Exception:
        macro["btc"] = pd.DataFrame(); macro["eth"] = pd.DataFrame()
    return macro

# --- News helpers (Yahoo Finance + Google News RSS fallback)
POLICY_NEG_KEYS = [
    "tariff", "trade war", "sanction", "ban", "export control", "retaliation", "retaliatory",
    "trump", "biden", "white house", "congress", "sec charges", "lawsuit", "antitrust",
    "probe", "investigation", "recall", "downgrade", "guidance cut", "secondary offering",
    "share offering", "convertible", "fraud", "accounting issue", "restatement", "delisting"
]

@st.cache_data(show_spinner=False, ttl=60*5)
def fetch_news_yf(ticker: str) -> pd.DataFrame:
    items = []
    try:
        n = yf.Ticker(ticker).news
        if n:
            for it in n[:30]:
                title = it.get("title", "")
                link = it.get("link", "")
                source = it.get("publisher", "")
                ts = it.get("providerPublishTime", None)
                published = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc) if ts else None
                items.append({"title": title, "source": source, "link": link, "published": published})
    except Exception:
        pass
    return pd.DataFrame(items)

@st.cache_data(show_spinner=False, ttl=60*5)
def fetch_news_google(ticker: str) -> pd.DataFrame:
    # Simple RSS parse (no extra deps)
    url = f"https://news.google.com/rss/search?q={quote_plus(ticker+' stock')}+when:7d&hl=en-US&gl=US&ceid=US:en"
    items = []
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r.content)
            for item in root.findall(".//item"):
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                pub = item.findtext("pubDate") or ""
                try:
                    published = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                except Exception:
                    published = None
                source = (item.find("source").text if item.find("source") is not None else "")
                items.append({"title": title, "source": source, "link": link, "published": published})
    except Exception:
        pass
    return pd.DataFrame(items)

@st.cache_data(show_spinner=False, ttl=60*5)
def get_news_with_sentiment(ticker: str) -> pd.DataFrame:
    df1 = fetch_news_yf(ticker)
    df2 = fetch_news_google(ticker)
    df = pd.concat([df1, df2], ignore_index=True)
    if df.empty:
        return df
    # Clean + dedupe
    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["source"] = df.get("source", pd.Series([""]*len(df)))
    df["published"] = pd.to_datetime(df.get("published"), errors="coerce", utc=True)
    df = df.sort_values("published", ascending=False)
    df = df.drop_duplicates(subset=["title"], keep="first")

    # Sentiment
    def _sent(x: str) -> float:
        if not x:
            return 0.0
        if _HAS_VADER:
            try:
                score = SentimentIntensityAnalyzer().polarity_scores(x)
                return float(score.get("compound", 0.0))
            except Exception:
                return 0.0
        # fallback mini-lexicon (very rough)
        pos = ["surge", "beat", "outperform", "upgrade", "soar", "growth", "record", "raise", "profit"]
        neg = ["miss", "downgrade", "fall", "plunge", "drop", "loss", "cut", "probe", "lawsuit", "ban", "tariff"]
        s = x.lower()
        return (sum(w in s for w in pos) - sum(w in s for w in neg)) / 5.0

    df["sentiment"] = df["title"].astype(str).apply(_sent)
    low_titles = df["title"].str.lower().fillna("")
    df["policy_risk"] = low_titles.apply(lambda s: any(k in s for k in POLICY_NEG_KEYS))
    # recency weight: 1.0 for <= 24h, then decay
    now = datetime.now(timezone.utc)
    hours = (now - df["published"]).dt.total_seconds().div(3600).fillna(999)
    df["recency_w"] = np.clip(np.exp(-hours/36.0), 0.05, 1.0)
    df["weighted_sent"] = df["sentiment"] * df["recency_w"]
    return df

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

# Indicator functions

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

def stochastic(high, low, close, n=14, k=3, d=3):
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    raw_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    K = raw_k.rolling(k).mean()
    D = K.rolling(d).mean()
    return K, D

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
# Load price data
# ==============================

data = fetch_prices(tickers, period)
if data is None or data.empty:
    st.error("No data retrieved. Check tickers or connection.")
    st.stop()

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

    # ----- Buy/Sell Scores
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

priority_cols = [
    "Ticker","Signal","BuyScore","SellScore","Price",
    "RSI14","%B","Stoch %K","Stoch %D","ADX14","Golden Cross","GCross Recent",
    "Ret 21D %","YTD %","EMA20","SMA200"
]
summary = summary[priority_cols + [c for c in summary.columns if c not in priority_cols]]

# ==============================
# Macro Regime (used later in probability)
# ==============================
macro = fetch_macro_series()

vix_series = macro.get("vix", pd.DataFrame())
if not vix_series.empty and "Close" in vix_series.columns:
    vix_close = vix_series["Close"].astype(float).dropna()
    if len(vix_close):
        vix_now = float(np.asarray(vix_close.iloc[-1]).ravel()[0])
        vix_pct = float(np.asarray(vix_close.rank(pct=True).iloc[-1]).ravel()[0])  # percentile in last 6m
    else:
        vix_now, vix_pct = np.nan, 0.5
else:
    vix_now, vix_pct = np.nan, 0.5

spy_series = macro.get("spy", pd.DataFrame())
spy_trend = 0.0
try:
    # Ensure we really have the Close column as a Series
    if isinstance(spy_series, pd.DataFrame) and ("Close" in spy_series.columns) and not spy_series.empty:
        spy_c = spy_series["Close"].astype(float).dropna()
        if spy_c.shape[0] > 50:
            spy_ema50 = ema(spy_c, 50)
            # Safely coerce to scalars even if objects come back as 0-d arrays/Series
            close_val = float(np.asarray(spy_c.iloc[-1]).ravel()[0])
            ema_val   = float(np.asarray(spy_ema50.iloc[-1]).ravel()[0])
            denom = ema_val if (np.isfinite(ema_val) and abs(ema_val) > 1e-6) else 1e-6
            spy_trend = float(np.tanh(((close_val - ema_val) / denom) * 10))
except Exception:
    # Keep default 0.0 if anything odd happens
    spy_trend = 0.0

btc = macro.get("btc", pd.DataFrame())
eth = macro.get("eth", pd.DataFrame())
crypto_24h = 0.0
try:
    if not btc.empty:
        btc_ret = (btc["Close"].iloc[-1] / btc["Close"].iloc[-24] - 1)
    else:
        btc_ret = 0.0
    if not eth.empty:
        eth_ret = (eth["Close"].iloc[-1] / eth["Close"].iloc[-24] - 1)
    else:
        eth_ret = 0.0
    crypto_24h = float((btc_ret + eth_ret) / 2.0)
except Exception:
    crypto_24h = 0.0

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
st.subheader("📊 Detailed Chart, News & Decision Panel")
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

# ---- Per-bar scores

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
    buy_th_use = st.session_state.get("_buy_th_cache", None) or buy_threshold
    sell_th_use = st.session_state.get("_sell_th_cache", None) or sell_threshold
    st.session_state["_buy_th_cache"] = buy_th_use
    st.session_state["_sell_th_cache"] = sell_th_use

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

entries, exits = generate_trades(p, buyS_s, sellS_s, buy_th_use, sell_th_use, atr_s,  atr_mult, rr_target)

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
rows_n = 1 + int(show_vol) + int(show_macd) + int(show_stoch) + int(show_adx) + 1  # price + optional panels + rsi
base = 0.55 if simple_view else 0.45
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
    rows=rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=row_heights, subplot_titles=titles
)

# ---- Row indices
row_idx = 1

# ---- Price row (candles + key overlays)
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
# 📰 News + Sentiment Table (selected ticker)
# ==============================
st.markdown("### 📰 News & Sentiment (last ~7 days)")
news_df = get_news_with_sentiment(selected)
if news_df.empty:
    st.info("No news fetched from Yahoo/Google RSS. (Network or ticker issue)")
else:
    show_cols = ["published", "source", "sentiment", "policy_risk", "title"]
    nd = news_df[show_cols].copy()
    nd = nd.rename(columns={"published":"Time (UTC)", "source":"Source", "sentiment":"Sent (-1..1)", "policy_risk":"PolicyRisk"})
    st.dataframe(nd, use_container_width=True, hide_index=True)

# ==============================
# 🎯 Decision Panel — Probability next move is UP
# ==============================
st.markdown("### 🎯 Decision Panel — Up/Down Likelihood (real-time)")

# 1) Technical edge from last bar
last_buy = float(buyS_s.iloc[-1]) if len(buyS_s) else 0.0
last_sell = float(sellS_s.iloc[-1]) if len(sellS_s) else 0.0
tech_edge = (last_buy - last_sell) / 100.0  # -1..1

# 2) News sentiment edge (weighted last 72h)
news_edge = 0.0
policy_hit = False
if not news_df.empty:
    recent = news_df[(pd.Timestamp.now(tz=timezone.utc) - news_df["published"]) <= pd.Timedelta(hours=72)]
    if recent.empty:
        recent = news_df.head(10)
    news_edge = float(np.tanh(recent["weighted_sent"].sum()))  # -1..1
    policy_hit = bool(recent["policy_risk"].any())

# 3) Macro regime: higher VIX subtracts, SPY up adds, crypto adds for crypto-sensitive stocks
macro_edge = 0.0
try:
    vix_val = float(np.asarray(vix_pct).ravel()[0])
    if np.isfinite(vix_val):
        macro_edge += -0.8 * float(np.maximum(0.0, vix_val - 0.5))  # if VIX high vs 6m, reduce odds
except Exception:
    pass
try:
    st_spy_trend = float(np.asarray(spy_trend).ravel()[0])
    if np.isfinite(st_spy_trend):
        macro_edge += 0.4 * st_spy_trend
except Exception:
    pass
# mild bonus if ticker is likely crypto-sensitive
if selected in {"HOOD","COIN","MSTR","MARA","RIOT","PYPL","SQ"}:
    macro_edge += 0.6 * np.tanh(crypto_24h * 10)

# 4) Volume impulse (today vs 20D avg) — if available for last bar
try:
    vol_ratio_last = float((v.iloc[-1] / v.rolling(20).mean().iloc[-1]))
    vol_edge = 0.3 * np.tanh((vol_ratio_last - 1.0) * 1.2)
except Exception:
    vol_edge = 0.0

# 5) Policy risk penalty if triggered in recent headlines
policy_penalty = -0.25 if policy_hit else 0.0

# Combine (logistic)
z = 1.6*tech_edge + 0.7*news_edge + 0.6*macro_edge + 0.3*vol_edge + policy_penalty
prob_up = 1.0 / (1.0 + math.exp(-z))  # 0..1
prob_up_pct = round(prob_up * 100.0, 1)

# ---- Gauge
fig_prob = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prob_up_pct,
    title = {"text": f"{selected} — Chance Up (next session)"},
    number = {"suffix":"%"},
    gauge = {
        "axis": {"range": [0, 100]},
        "bar": {"thickness": 0.25},
        "steps": [
            {"range": [0, 40],  "color": "#ffcccc"},
            {"range": [40, 60], "color": "#fff4cc"},
            {"range": [60, 100],"color": "#d8f5d0"}
        ],
        "threshold": {"line": {"width": 4}, "thickness": 0.8, "value": prob_up_pct}
    }
))
fig_prob.update_layout(height=240, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_prob, use_container_width=True)

# ---- Reasons panel
bull_pts, bear_pts = [], []
if tech_edge > 0: bull_pts.append("Technical scores favor BUY over SELL")
else: bear_pts.append("Technical scores lean to SELL/Trim")

if news_edge > 0.05: bull_pts.append("News sentiment positive (recent headlines)")
elif news_edge < -0.05: bear_pts.append("News sentiment negative (recent headlines)")
else: pass

if spy_trend > 0.1: bull_pts.append("SPY above trend (risk-on)")
elif spy_trend < -0.1: bear_pts.append("SPY below trend (risk-off)")

try:
    vix_flag_val = float(np.asarray(vix_pct).ravel()[0])
    if np.isfinite(vix_flag_val) and vix_flag_val > 0.7:
        bear_pts.append("VIX elevated vs 6m (volatility risk)")
except Exception:
    pass

if selected in {"HOOD","COIN","MSTR","MARA","RIOT","PYPL","SQ"}:
    if crypto_24h > 0: bull_pts.append("Crypto 24h up (supportive for flow-sensitive names)")
    elif crypto_24h < 0: bear_pts.append("Crypto 24h down (headwind)")

if policy_hit: bear_pts.append("Policy/tariff/regulatory headline detected — caution")

colL, colR = st.columns(2)
with colL:
    st.markdown("**Bullish factors**")
    if bull_pts:
        st.markdown("\n".join([f"- {x}" for x in bull_pts]))
    else:
        st.markdown("- (none strong)")
with colR:
    st.markdown("**Bearish factors**")
    if bear_pts:
        st.markdown("\n".join([f"- {x}" for x in bear_pts]))
    else:
        st.markdown("- (none strong)")

# ==============================
# Tips (short)
# ==============================
with st.expander("💡 Quick tips"):
    st.markdown(
        "- **BUY**: BuyScore rising, MACD hist flips positive, %B ~ 0–0.35, RSI below threshold, Stoch %K < 20, ADX>25 helpful, Vol Ratio > 1 if possible.\n"
        "- **SELL/Trim**: SellScore high, %B ~ 0.75–1.0, RSI > 65–70, Stoch %K > 80, MACD hist falling/negative, near 52w high.\n"
        "- **Probability gauge** is a blend of **technical**, **news sentiment**, **macro**, **volume**, and **policy risk**.\n"
        "- Quiet sessions → **Lenient mode** or **Adaptive thresholds**.\n"
        "- Always cross-check with broader context (earnings, news)."
    )

# ==============================
# Footer — safety note
# ==============================
st.caption("This tool is educational only. Markets can move fast; probabilities are not guarantees.")
