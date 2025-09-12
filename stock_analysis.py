# -*- coding: utf-8 -*-
"""
Market Decision Dashboard — Plotly Dash (pure local)
- Single-file app with:
  • Detailed ticker view (candles + overlays)
  • Single-ticker probability gauge + reasons + Kelly (capped)
  • Multi-tile gauges (up to 12 tickers) + summary
  • News sentiment (VADER with fallback)
  • Macro blend (VIX percentile, SPY trend, BTC/ETH), policy-risk penalty
Run:
  python app.py
Open:
  http://localhost:8080
"""
import os, math
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# ---------- config ----------
DEFAULT_TICKERS = ["HOOD", "NVDA", "AMD", "META", "AMAT", "TSLA", "AMZA", "MSFT"]
ALIAS = {"NVDIA": "NVDA"}
POLICY_NEG_KEYS = [
    "tariff","trade war","sanction","ban","export control","retaliation","retaliatory",
    "trump","biden","white house","congress","sec charges","lawsuit","antitrust",
    "probe","investigation","recall","downgrade","guidance cut","secondary offering",
    "share offering","convertible","fraud","accounting issue","restatement","delisting"
]

# ---------- indicators ----------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    val = 100 - (100 / (1 + rs))
    return val.fillna(50)

def macd(series: pd.Series):
    m = ema(series, 12) - ema(series, 26)
    s = ema(m, 9)
    h = m - s
    return m, s, h

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower).replace(0, np.nan)
    pb = (series - lower) / width
    return ma, upper, lower, pb

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def pct_from_52wk_ext(series: pd.Series, window: int = 252):
    rh = series.rolling(window).max()
    rl = series.rolling(window).min()
    dist_low = (series - rl) / rl.replace(0, np.nan) * 100
    dist_high = (rh - series) / rh.replace(0, np.nan) * 100
    return dist_low, dist_high

def perbar_scores(price: pd.Series, rsi_s: pd.Series, pb_s: pd.Series,
                  hist_s: pd.Series, vol_s: pd.Series,
                  rsi_buy_th: int = 55, rsi_sell_th: int = 65,
                  bb_buy_pct: float = 0.35, bb_sell_pct: float = 0.75,
                  use_volume_gate: bool = False, vol_ratio_min: float = 0.4):
    volr = (vol_s / vol_s.rolling(20).mean()).replace([np.inf, -np.inf], np.nan)
    dlow, dhigh = pct_from_52wk_ext(price)
    buyS = pd.Series(0.0, index=price.index, dtype=float)
    sellS = pd.Series(0.0, index=price.index, dtype=float)
    for i in range(len(price)):
        rsiv = rsi_s.iloc[i]; pbv = pb_s.iloc[i]; hv = hist_s.iloc[i]; vr = volr.iloc[i]
        dl = dlow.iloc[i]; dh = dhigh.iloc[i]
        bs = 0.0
        if pd.notna(rsiv): bs += max(0, (rsi_buy_th - rsiv)) / max(1, rsi_buy_th) * 35
        if pd.notna(pbv):  bs += max(0.0, (bb_buy_pct - pbv)) / max(0.001, bb_buy_pct) * 25
        if pd.notna(hv) and (hv > 0): bs += 5.0
        if pd.notna(vr):
            if vr > 1: bs += min(15.0, (vr - 1)) * 15.0
            elif use_volume_gate and (vr < vol_ratio_min): bs -= 8.0
        if pd.notna(dl) and (dl < 20): bs += 8.0
        buyS.iloc[i] = min(100.0, max(0.0, bs))
        ss = 0.0
        if pd.notna(rsiv): ss += max(0, (rsiv - rsi_sell_th)) / max(1, (100 - rsi_sell_th)) * 35
        if pd.notna(pbv):  ss += max(0.0, (pbv - bb_sell_pct)) / max(0.001, (1 - bb_sell_pct)) * 25
        if pd.notna(hv) and (hv < 0): ss += 5.0
        if pd.notna(dh) and (dh < 12): ss += 10.0
        sellS.iloc[i] = min(100.0, max(0.0, ss))
    return buyS, sellS

# ---------- data & news ----------
def fetch_prices(tickers: List[str], period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    return yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False)

def get_field(df: pd.DataFrame, field: str, tickers: List[str]) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        have = [t for t in tickers if t in df[field].columns]
        return df[field][have].copy()
    else:
        t = tickers[0]
        out = df[[field]].copy()
        out.columns = [t]
        return out

def fetch_news_yf(ticker: str) -> pd.DataFrame:
    items = []
    try:
        n = yf.Ticker(ticker).news
        if n:
            for it in n[:30]:
                title = it.get("title", ""); link = it.get("link", "")
                source = it.get("publisher", ""); ts = it.get("providerPublishTime", None)
                published = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc) if ts else None
                items.append({"title": title, "source": source, "link": link, "published": published})
    except Exception:
        pass
    return pd.DataFrame(items)

def fetch_news_google(ticker: str) -> pd.DataFrame:
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(ticker+' stock')}+when:7d&hl=en-US&gl=US&ceid=US:en"
    items = []
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r.content)
            for item in root.findall(".//item"):
                title = item.findtext("title") or ""
                link  = item.findtext("link") or ""
                pub   = item.findtext("pubDate") or ""
                try:
                    published = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                except Exception:
                    published = None
                source = (item.find("source").text if item.find("source") is not None else "")
                items.append({"title": title, "source": source, "link": link, "published": published})
    except Exception:
        pass
    return pd.DataFrame(items)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

def get_news_with_sentiment(ticker: str) -> pd.DataFrame:
    df = pd.concat([fetch_news_yf(ticker), fetch_news_google(ticker)], ignore_index=True)
    if df.empty: return df
    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["source"] = df.get("source", pd.Series([""]*len(df)))
    df["published"] = pd.to_datetime(df.get("published"), errors="coerce", utc=True)
    df = df.sort_values("published", ascending=False).drop_duplicates(subset=["title"], keep="first")
    def _sent(x: str) -> float:
        if not x: return 0.0
        if _VADER is not None:
            try: return float(_VADER.polarity_scores(x).get("compound", 0.0))
            except Exception: return 0.0
        pos = ["surge","beat","outperform","upgrade","soar","growth","record","raise","profit"]
        neg = ["miss","downgrade","fall","plunge","drop","loss","cut","probe","lawsuit","ban","tariff"]
        s = x.lower(); return (sum(w in s for w in pos) - sum(w in s for w in neg)) / 5.0
    df["sentiment"] = df["title"].astype(str).apply(_sent)
    low = df["title"].str.lower().fillna("")
    df["policy_risk"] = low.apply(lambda s: any(k in s for k in POLICY_NEG_KEYS))
    now = datetime.now(timezone.utc)
    hrs = (now - df["published"]).dt.total_seconds().div(3600).fillna(999)
    df["recency_w"] = np.clip(np.exp(-hrs/36.0), 0.05, 1.0)
    df["weighted_sent"] = df["sentiment"] * df["recency_w"]
    return df

# ---------- macro ----------
def fetch_macro():
    out: Dict[str, Any] = {}
    try: out["vix"] = yf.download("^VIX", period="6mo", interval="1d", progress=False)
    except Exception: out["vix"] = pd.DataFrame()
    try: out["spy"] = yf.download("SPY", period="3mo", interval="1d", progress=False)
    except Exception: out["spy"] = pd.DataFrame()
    try:
        out["btc"] = yf.download("BTC-USD", period="7d", interval="1h", progress=False)
        out["eth"] = yf.download("ETH-USD", period="7d", interval="1h", progress=False)
    except Exception:
        out["btc"] = pd.DataFrame(); out["eth"] = pd.DataFrame()
    return out

# ---------- probability ----------
def compute_probability_and_reasons(ticker: str, period: str = "3mo",
                                    rsi_buy_th: int = 55, rsi_sell_th: int = 65,
                                    bb_buy_pct: float = 0.35, bb_sell_pct: float = 0.75,
                                    rr_target: float = 1.5):
    t = ALIAS.get(ticker, ticker)
    df = fetch_prices([t], period=period, interval="1d")
    if df is None or df.empty: return 0.5, {"error": f"No price data for {t}"}

    def _field(field: str) -> pd.Series:
        got = get_field(df, field, [t])
        return got[t].dropna()

    # ✅ correct conditional
    p = _field("Adj Close") if (isinstance(df.columns, pd.MultiIndex) and "Adj Close" in df.columns.get_level_values(0)) else _field("Close")
    o = _field("Open").reindex(p.index)
    h = _field("High").reindex(p.index)
    l = _field("Low").reindex(p.index)
    v = _field("Volume").reindex(p.index).fillna(0)
    if p.empty or len(p) < 30: return 0.5, {"error": f"Insufficient data for {t}"}

    ema20 = ema(p, 20); ema50 = ema(p, 50); sma200 = p.rolling(200).mean()
    _, upBB, loBB, pb = bollinger(p, 20, 2)
    _, _, hist = macd(p)
    rsi_s = rsi(p, 14)

    buyS, sellS = perbar_scores(p, rsi_s, pb, hist, v,
                                rsi_buy_th=rsi_buy_th, rsi_sell_th=rsi_sell_th,
                                bb_buy_pct=bb_buy_pct, bb_sell_pct=bb_sell_pct)

    last_buy = float(buyS.iloc[-1]) if len(buyS) else 0.0
    last_sell = float(sellS.iloc[-1]) if len(sellS) else 0.0
    tech_edge = (last_buy - last_sell) / 100.0

    df_news = get_news_with_sentiment(t)
    news_edge = 0.0; policy_hit = False
    if df_news is not None and not df_news.empty:
        recent = df_news[(pd.Timestamp.now(tz=timezone.utc) - df_news["published"]) <= pd.Timedelta(hours=72)]
        if recent.empty: recent = df_news.head(10)
        news_edge = float(np.tanh(recent["weighted_sent"].sum()))
        policy_hit = bool(recent["policy_risk"].any())

    macro = fetch_macro()
    vix_series = macro.get("vix", pd.DataFrame())
    vix_pct = 0.5
    if not vix_series.empty and "Close" in vix_series.columns:
        vix_close = vix_series["Close"].astype(float).dropna()
        if len(vix_close): vix_pct = float(np.asarray(vix_close.rank(pct=True).iloc[-1]).ravel()[0])
    macro_edge = -0.8 * float(np.maximum(0.0, vix_pct - 0.5))

    spy_series = macro.get("spy", pd.DataFrame())
    if not spy_series.empty and "Close" in spy_series.columns and spy_series["Close"].shape[0] > 50:
        spy_c = spy_series["Close"].astype(float).dropna()
        spy_ema50 = ema(spy_c, 50)
        close_val = float(np.asarray(spy_c.iloc[-1]).ravel()[0])
        ema_val = float(np.asarray(spy_ema50.iloc[-1]).ravel()[0])
        denom = ema_val if (np.isfinite(ema_val) and abs(ema_val) > 1e-6) else 1e-6
        spy_trend = float(np.tanh(((close_val - ema_val) / denom) * 10))
        macro_edge += 0.4 * spy_trend

    btc = macro.get("btc", pd.DataFrame()); eth = macro.get("eth", pd.DataFrame())
    crypto_24h = 0.0
    try:
        btc_ret = float(btc["Close"].iloc[-1] / btc["Close"].iloc[-24] - 1) if not btc.empty else 0.0
        eth_ret = float(eth["Close"].iloc[-1] / eth["Close"].iloc[-24] - 1) if not eth.empty else 0.0
        crypto_24h = (btc_ret + eth_ret) / 2.0
    except Exception:
        crypto_24h = 0.0
    if t in {"HOOD","COIN","MSTR","MARA","RIOT","PYPL","SQ"}:
        macro_edge += 0.6 * np.tanh(crypto_24h * 10)

    try:
        vol_ratio_last = float((v.iloc[-1] / v.rolling(20).mean().iloc[-1]))
        vol_edge = 0.3 * np.tanh((vol_ratio_last - 1.0) * 1.2)
    except Exception:
        vol_edge = 0.0

    policy_penalty = -0.25 if policy_hit else 0.0
    z = 1.6*tech_edge + 0.7*news_edge + 0.6*macro_edge + 0.3*vol_edge + policy_penalty
    prob_up = 1.0 / (1.0 + math.exp(-z))
    prob_pct = float(round(prob_up * 100.0, 1))

    bull, bear = [], []
    bull.append("Technical scores favor BUY over SELL") if tech_edge > 0 else bear.append("Technical scores lean to SELL/Trim")
    if news_edge > 0.05: bull.append("News sentiment positive (recent)")
    elif news_edge < -0.05: bear.append("News sentiment negative (recent)")
    if vix_pct > 0.7: bear.append("VIX elevated vs 6m (volatility risk)")
    if t in {"HOOD","COIN","MSTR","MARA","RIOT","PYPL","SQ"}:
        if crypto_24h > 0: bull.append("Crypto 24h up (supportive)")
        elif crypto_24h < 0: bear.append("Crypto 24h down (headwind)")
    if policy_hit: bear.append("Policy/tariff/regulatory headline detected — caution")

    try:
        b_rr = float(rr_target) if np.isfinite(rr_target) and rr_target > 0 else 1.0
        kelly = (b_rr*prob_up - (1.0 - prob_up)) / b_rr
        kelly_capped = float(np.clip(kelly, 0.0, 0.20))
    except Exception:
        kelly_capped = 0.0

    series = {"p": p, "o": o, "h": h, "l": l, "ema20": ema20, "ema50": ema50, "sma200": sma200, "upBB": upBB, "loBB": loBB}
    meta = {"ticker": t, "prob_pct": prob_pct, "bull": bull, "bear": bear, "kelly": float(round(kelly_capped*100, 1))}
    return prob_up, {"meta": meta, "series": series}

# ---------- charts ----------
def price_chart(series: Dict[str, pd.Series], ticker: str) -> go.Figure:
    p = series["p"]; o = series["o"]; h = series["h"]; l = series["l"]
    ema20 = series["ema20"]; ema50 = series["ema50"]; sma200 = series["sma200"]
    upBB = series["upBB"]; loBB = series["loBB"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=p.index, open=o, high=h, low=l, close=p, name="Price"))
    fig.add_trace(go.Scatter(x=p.index, y=ema20, name="EMA20", line=dict(width=1.6)))
    fig.add_trace(go.Scatter(x=p.index, y=ema50, name="EMA50", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=p.index, y=sma200, name="SMA200", line=dict(dash="dash", width=1.2)))
    fig.add_trace(go.Scatter(x=p.index, y=upBB, name="Upper BB", line=dict(dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=p.index, y=loBB, name="Lower BB", line=dict(dash="dot", width=1)))
    fig.update_layout(title=f"{ticker} — Price & Signals", xaxis_rangeslider_visible=False,
                      height=520, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def probability_gauge(title: str, pct: float, mini: bool = False) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct, title={"text": title},
        number={"suffix": "%", "font": {"size": 26 if mini else 36}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"thickness": 0.25},
               "steps": [{"range": [0, 40], "color": "#ffe6e6"},
                         {"range": [40, 60], "color": "#fff6d5"},
                         {"range": [60, 100], "color": "#e6f7e6"}]}
    ))
    fig.update_layout(height=180 if mini else 240, margin=dict(l=10, r=10, t=30, b=6))
    return fig

# ---------- dash app ----------
external_stylesheets = [dbc.themes.FLATLY]
app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

def control_panel():
    return dbc.Card([
        dbc.CardHeader("Controls"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Detail ticker"),
                         dcc.Dropdown(id="detail-ticker",
                                      options=[{"label": t, "value": t} for t in DEFAULT_TICKERS],
                                      value=DEFAULT_TICKERS[0], clearable=False)], md=4),
                dbc.Col([html.Label("Tiles (up to 12)"),
                         dcc.Dropdown(id="tile-tickers",
                                      options=[{"label": t, "value": t} for t in sorted(set(DEFAULT_TICKERS))],
                                      value=DEFAULT_TICKERS, multi=True)], md=8),
            ], className="g-2"),
            html.Hr(),
            dbc.Button("Refresh now", id="refresh-btn", color="primary")
        ])
    ], className="mb-3")

app.layout = dbc.Container([
    html.H2("📈 Market Decision Dashboard — Dash (Local)"),
    html.Div("Runs entirely on your laptop. No cloud, no public URL."),
    html.Hr(),
    control_panel(),
    dbc.Row([
        dbc.Col(dcc.Graph(id="detail-chart"), md=8),
        dbc.Col([dcc.Graph(id="detail-gauge"), html.Div(id="detail-reasons", className="mt-2")], md=4)
    ], className="g-3"),
    html.Hr(),
    html.H4("🧩 Multi-Tile Probability Dashboard"),
    dbc.Row(id="tiles-row", className="g-3"),
    html.Div(id="tiles-summary", className="mt-3"),
], fluid=True)

# callbacks
@app.callback(
    [Output("detail-chart", "figure"),
     Output("detail-gauge", "figure"),
     Output("detail-reasons", "children")],
    [Input("detail-ticker", "value"), Input("refresh-btn", "n_clicks")]
)
def update_detail(ticker, _n):
    if not ticker: ticker = DEFAULT_TICKERS[0]
    ticker = ALIAS.get(ticker, ticker)
    _prob, out = compute_probability_and_reasons(ticker)
    meta = out.get("meta", {}); series = out.get("series", {})
    fig_chart = price_chart(series, meta.get("ticker", ticker)) if series else go.Figure()
    gauge = probability_gauge(f"{meta.get('ticker', ticker)} — Chance Up (next session)", meta.get("prob_pct", 50.0))
    bull = meta.get("bull", []); bear = meta.get("bear", []); kelly = meta.get("kelly", 0.0)
    reasons = [
        html.Strong("Bullish"), html.Ul([html.Li(x) for x in bull]) if bull else html.Div("(none)"),
        html.Strong("Bearish"), html.Ul([html.Li(x) for x in bear]) if bear else html.Div("(none)"),
        html.Div(f"Suggested sizing (Kelly capped): {kelly:.1f}% of capital (educational)",
                 style={"fontSize": "0.9rem", "color": "#555"})
    ]
    return fig_chart, gauge, reasons

@app.callback(
    [Output("tiles-row", "children"), Output("tiles-summary", "children")],
    [Input("tile-tickers", "value"), Input("refresh-btn", "n_clicks")]
)
def update_tiles(tickers, _n):
    if not tickers: tickers = DEFAULT_TICKERS
    tickers = [ALIAS.get(t, t) for t in tickers][:12]
    cards, rows = [], []
    for t in tickers:
        try:
            _, out = compute_probability_and_reasons(t)
            meta = out.get("meta", {}); pct = meta.get("prob_pct", 50.0); kelly = meta.get("kelly", 0.0)
            fig = probability_gauge(meta.get("ticker", t), pct, mini=True)
            cards.append(dbc.Col(dcc.Graph(figure=fig), md=3, sm=6))
            rows.append({"Ticker": meta.get("ticker", t), "ProbUp %": pct,
                         "Kelly % (cap 20)": kelly,
                         "Suggested": ("Buy bias" if pct >= 60 else ("Wait/Neutral" if pct >= 40 else "Avoid/Sell bias"))})
        except Exception:
            fig = probability_gauge(t, 50.0, mini=True)
            cards.append(dbc.Col(dcc.Graph(figure=fig), md=3, sm=6))
    if rows:
        df_sum = pd.DataFrame(rows).sort_values("ProbUp %", ascending=False)
        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_sum.columns],
            data=df_sum.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "fontSize": "14px"}
        )
        summary = dbc.Card([dbc.CardHeader("Summary (mini-gauges)"), dbc.CardBody(table)])
    else:
        summary = html.Div("No data")
    return cards, summary

if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port=8080, debug=False)
