##############################################################################################

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import random
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

##############################################################################################

# Configure page layout
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# App Title
st.title("Stock Performance Dashboard")

# — Refresh Button —
if st.button("🔄 Refresh Data"):
    st.experimental_set_query_params(refresh=random.random())

# — User Inputs —
tickers_input = st.text_area(
    "Enter stock tickers (comma-separated):",
    value="AAPL, MSFT, GOOG, GOOGL, NVDA, AMZN, TSLA, FB, BABA, JPM, V, TSM, HOOD, NFLX, AMAT, META, AMD, INTC, QTUM, SPY, VOO, NOBL"
)
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

period = st.selectbox(
    "Select historical window:",
    options=["1w", "2w", "3w", "1m", "2m", "3m", "1y", "2y", "3y", "5y"],
    index=5  # default 3m
)

top_n = st.slider(
    "Number of top stocks to display:",
    min_value=5, max_value=50, value=10, step=5
)

if not tickers:
    st.error("Please enter at least one ticker symbol.")
    st.stop()

##############################################################################################

# — Updated Guidelines Sidebar —
st.sidebar.header("📋 Parameter Guidelines")
st.sidebar.markdown(
    """
**Our goal:** Buy low (oversold) and Sell high (overbought)

1. **RSI14**  
   - < 30 → Oversold (✅ BUY signal)  
   - 30–70 → Neutral  
   - > 70 → Overbought (⚠️ SELL/AVOID)  

2. **MACD Histogram**  
   - Positive → Bullish momentum (✅ HOLD or ADD)  
   - Turns Negative → Momentum fading (⚠️ SELL signal)  

3. **Vol Ratio** (Today’s Volume ÷ 20-day Avg)  
   - > 2 → Very strong buying interest (✅)  
   - 1–2 → Above average interest (✔️)  
   - < 1 → Low interest (⚠️)  

4. **1M %** (21-day % change) – Higher = stronger upward trend (✅)  
5. **Weekly %** (5-day % change) – Higher = recent strength (✅)  
6. **Daily %** – Positive = price up today (✅)  

7. **Above EMA20**  
   - Yes → Price > 20-day EMA (✅ Bullish)  
   - No  → Price < EMA20 (⚠️ Bearish)  

8. **Above 50DMA**  
   - Yes → Price > 50-day MA (✅ Bullish)  
   - No  → Price < 50DMA (⚠️ Bearish)  

9. **Bollinger Bands**  
   - Near Lower Band → Possible buy zone (✅)  
   - Near Upper Band → Price overextended (⚠️ sell or wait)  
"""
)

##############################################################################################

# — Fetch Data with custom periods —
def fetch_data(tickers, window):
    now = datetime.now()
    if window.endswith('w'):
        weeks = int(window[:-1])
        start = now - pd.DateOffset(weeks=weeks)
        return yf.download(tickers, start=start, end=now, interval="1d")
    if window.endswith('m') and window != "1m":
        months = int(window[:-1])
        start = now - pd.DateOffset(months=months)
        return yf.download(tickers, start=start, end=now, interval="1d")
    if window == "1m":
        return yf.download(tickers, period="1mo", interval="1d")
    # for years, use built-in period
    return yf.download(tickers, period=window, interval="1d")

try:
    data = fetch_data(tickers, period)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if data is None or data.empty:
    st.error("No data retrieved. Check tickers or your connection.")
    st.stop()

# — Prepare price & volume frames —
price_field = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
vol_field   = 'Volume'


##############################################################################################

# Extract price and volume data
if isinstance(data.columns, pd.MultiIndex):
    price_df  = data[price_field]
    volume_df = data[vol_field]
else:
    price_df  = data[[price_field]].rename(columns={price_field: tickers[0]})
    volume_df = data[[vol_field]].rename(columns={vol_field: tickers[0]})

# — Compute Metrics —
def compute_metrics(prices, vols):
    latest = prices.iloc[-1]
    prev   = prices.iloc[-2] if len(prices) >= 2 else np.nan
    daily  = (latest - prev)/prev*100 if not np.isnan(prev) else np.nan

    def pctchg(n): return (latest/prices.iloc[-n-1]-1)*100 if len(prices)>n else np.nan
    weekly = pctchg(5)
    monthly= pctchg(21)

    ema20      = prices.ewm(span=20).mean().iloc[-1]
    above_ema20= latest>ema20
    dma50      = prices.rolling(50).mean().iloc[-1]
    above_dma50= latest>dma50

    delta = prices.diff().dropna()
    gain  = delta.where(delta>0,0.0).rolling(14).mean().iloc[-1]
    loss  = -delta.where(delta<0,0.0).rolling(14).mean().iloc[-1]
    rsi14 = 100-(100/(1+gain/loss)) if loss!=0 else 100

    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd_line = ema12-ema26
    signal_line= macd_line.ewm(span=9).mean()
    hist      = macd_line.iloc[-1]-signal_line.iloc[-1]

    sma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    upper_bb = sma20+2*std20
    lower_bb = sma20-2*std20

    avg_vol   = vols.rolling(20).mean().iloc[-1]
    vol_ratio = vols.iloc[-1]/avg_vol if avg_vol>0 else np.nan

    notes=[]
    if rsi14<30: notes.append("Oversold")
    if rsi14>70: notes.append("Overbought")
    if vol_ratio>2: notes.append("HighVol")
    sig="; ".join(notes)

    return {
        'Price': round(latest,2),
        'RSI14': round(rsi14,1),
        'Hist': round(hist,2),
        'Vol Ratio': round(vol_ratio,2),
        '1M %': round(monthly,2),
        'Weekly %': round(weekly,2),
        'Daily %': round(daily,2),
        'Above EMA20': 'Yes' if above_ema20 else 'No',
        'Above 50DMA': 'Yes' if above_dma50 else 'No',
        'UpperBB': round(upper_bb.iloc[-1],2),
        'LowerBB': round(lower_bb.iloc[-1],2),
        'Signals': sig
    }

metrics=[]
for t in price_df.columns:
    p=price_df[t].dropna()
    v=volume_df[t].dropna()
    if not p.empty:
        d=compute_metrics(p,v); d['Ticker']=t
        metrics.append(d)

df=pd.DataFrame(metrics)

# — Order dashboard elements for best decision flow —
# 1) Top Performers
sort_by = st.selectbox("Sort by:", options=[c for c in df.columns if c not in ['Ticker','Signals']], index=0)
df_sorted = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
df_top    = df_sorted.head(top_n)

st.subheader(f"🏆 Top {top_n} by {sort_by}")
fig_top, ax = plt.subplots(figsize=(10,4))
ax.bar(df_top['Ticker'], df_top[sort_by])
ax.set_ylabel(sort_by); ax.set_title("Top Performers")
st.pyplot(fig_top)

# 2) Parameter Bar Chart
st.subheader("📊 Top Stocks Metrics Overview")
to_plot = st.multiselect("Metrics to visualize:", ['RSI14','Hist','Vol Ratio','1M %','Weekly %','Daily %'], default=['RSI14','1M %','Vol Ratio'])
if to_plot:
    st.bar_chart(df_top.set_index('Ticker')[to_plot])

# 3) Metrics Table
cols = ['Ticker','Price','RSI14','Hist','Vol Ratio','1M %','Weekly %','Daily %',
        'Above EMA20','Above 50DMA','UpperBB','LowerBB','Signals']
st.subheader("🔎 Stock Metrics Table")
st.dataframe(df_sorted[cols])


# --- Detailed Plots per Selected Stock (Interactive, Dynamic Window) ---
selected = st.selectbox("Detailed charts for:", price_df.columns.tolist())

# Determine cutoff date based on selected 'period' (e.g. "1w","2m","1y", etc.)
latest_date = price_df.index.max()
if period.endswith('w'):
    cutoff = latest_date - pd.DateOffset(weeks=int(period[:-1]))
elif period.endswith('m'):
    cutoff = latest_date - pd.DateOffset(months=int(period[:-1]))
elif period.endswith('y'):
    cutoff = latest_date - pd.DateOffset(years=int(period[:-1]))
else:
    cutoff = price_df.index.min()

# Slice data to the cutoff
prices = price_df[selected].dropna().loc[cutoff:]
vols   = volume_df[selected].dropna().loc[cutoff:]

# Compute indicators on this window
ema20      = prices.ewm(span=20, adjust=False).mean()
sma20      = prices.rolling(20).mean()
std20      = prices.rolling(20).std()
upper_bb   = sma20 + 2 * std20
lower_bb   = sma20 - 2 * std20
ema12      = prices.ewm(span=12, adjust=False).mean()
ema26      = prices.ewm(span=26, adjust=False).mean()
macd_line  = ema12 - ema26
signal_line= macd_line.ewm(span=9, adjust=False).mean()
hist       = macd_line - signal_line
rsi        = 100 - (100 / (
                 1 + (
                     prices.diff().clip(lower=0).rolling(14).mean() /
                     -prices.diff().clip(upper=0).rolling(14).mean()
                 )
             ))

# Create 3-row subplot with shared x-axis
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        f"{selected} Price & Bollinger Bands",
        f"{selected} MACD",
        f"{selected} RSI14"
    )
)

# Row 1: Price & Bands
fig.add_trace(go.Scatter(x=prices.index, y=prices,    name='Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=ema20.index,  y=ema20,     name='EMA20'), row=1, col=1)
fig.add_trace(go.Scatter(x=upper_bb.index, y=upper_bb, name='Upper BB', line=dict(dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=lower_bb.index, y=lower_bb, name='Lower BB', line=dict(dash='dash')), row=1, col=1)

# Row 2: MACD + Histogram
fig.add_trace(go.Scatter(x=macd_line.index,  y=macd_line,   name='MACD'), row=2, col=1)
fig.add_trace(go.Scatter(x=signal_line.index,y=signal_line, name='Signal'), row=2, col=1)
fig.add_trace(go.Bar(    x=hist.index,      y=hist,         name='Histogram', opacity=0.6), row=2, col=1)

# Row 3: RSI with thresholds
fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI14'), row=3, col=1)
fig.add_trace(go.Scatter(x=rsi.index, y=[70]*len(rsi), name='Overbought (70)',
                         line=dict(dash='dash', color='red')), row=3, col=1)
fig.add_trace(go.Scatter(x=rsi.index, y=[30]*len(rsi), name='Oversold (30)',
                         line=dict(dash='dash', color='green')), row=3, col=1)

# Layout tweaks
fig.update_layout(
    height=900,
    title_text=f"{selected} Technical Indicators ({period} window)",
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)