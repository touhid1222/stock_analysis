# -*- coding: utf-8 -*-
"""
marketdash.utils
Shared utilities for the Market Decision Dashboard:
- yfinance fetching with retry
- indicators (EMA, RSI, MACD, Bollinger, ATR)
- news fetch + sentiment + recency weighting
- macro fetch (VIX, SPY)
- intraday 5m drift
- probability blend + ATR position sizing
- chart builders (price+RSI, gauge) and placeholder figure
"""

# utils.py

import math, time
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional sentiment; safe fallback if not present
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None


# ---------------- Small general helpers ----------------
def retry(fn, tries: int = 2, sleep: float = 0.8, default=None):
    for i in range(max(1, tries)):
        try:
            return fn()
        except Exception:
            if i < tries - 1:
                time.sleep(sleep)
            else:
                return default

def empty_figure(title: str = "No data", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.add_annotation(text=title, x=0.5, y=0.5, showarrow=False)
    return fig


# ---------------- Technical indicators ----------------
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
    return ma, upper, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------------- Data helpers (with retry) ----------------
def fetch_prices(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    def _call():
        return yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    df = retry(_call, tries=2, sleep=0.7, default=pd.DataFrame())
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def fetch_prices_multi(tickers: List[str], period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    def _call():
        return yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False)
    df = retry(_call, tries=2, sleep=0.7, default=pd.DataFrame())
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


# ---------------- News & Sentiment ----------------
def fetch_news_yf(ticker: str) -> pd.DataFrame:
    items = []
    try:
        n = yf.Ticker(ticker).news
        if n:
            for it in n[:40]:
                title = it.get("title", ""); link = it.get("link", "")
                src = it.get("publisher", ""); ts = it.get("providerPublishTime", None)
                published = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc) if ts else None
                items.append({"title": title, "source": src, "link": link, "published": published})
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
                src = (item.find("source").text if item.find("source") is not None else "")
                items.append({"title": title, "source": src, "link": link, "published": published})
    except Exception:
        pass
    return pd.DataFrame(items)

def get_news_with_sentiment(ticker: str) -> pd.DataFrame:
    df = pd.concat([fetch_news_yf(ticker), fetch_news_google(ticker)], ignore_index=True)
    if df.empty:
        return df
    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    if "source" not in df.columns:
        df["source"] = ""
    df["published"] = pd.to_datetime(df.get("published"), errors="coerce", utc=True)
    df = df.sort_values("published", ascending=False).drop_duplicates(subset=["title"], keep="first")

    def _sent(x: str) -> float:
        if not x:
            return 0.0
        if _VADER is not None:
            try:
                return float(_VADER.polarity_scores(x).get("compound", 0.0))
            except Exception:
                return 0.0
        pos = ["surge","beat","outperform","upgrade","soar","growth","record","raise","profit"]
        neg = ["miss","downgrade","fall","plunge","drop","loss","cut","probe","lawsuit","ban","tariff"]
        s = x.lower()
        return (sum(w in s for w in pos) - sum(w in s for w in neg)) / 5.0

    df["sentiment"] = df["title"].astype(str).apply(_sent)
    now = datetime.now(timezone.utc)
    hrs = (now - df["published"]).dt.total_seconds().div(3600).fillna(999)
    fast = np.exp(-np.clip(hrs, 0, 6)/2.0)       # 0-6h strongest
    mid  = np.exp(-np.clip(hrs-6, 0, 18)/6.0)    # 6-24h
    slow = np.exp(-np.clip(hrs-24, 0, 48)/18.0)  # 24-72h
    df["recency_w"] = np.clip(0.6*fast + 0.3*mid + 0.1*slow, 0.05, 1.0)
    df["weighted_sent"] = df["sentiment"] * df["recency_w"]
    return df


# ---------------- Macro ----------------
def fetch_macro() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    out["vix"] = retry(lambda: yf.download("^VIX", period="6mo", interval="1d", auto_adjust=False, progress=False),
                       tries=2, default=pd.DataFrame())
    out["spy"] = retry(lambda: yf.download("SPY", period="12mo", interval="1d", auto_adjust=False, progress=False),
                       tries=2, default=pd.DataFrame())
    return out


# ---------------- Intraday helpers ----------------
def intraday_drift_5m(ticker: str) -> Tuple[Optional[float], Optional[float], bool]:
    try:
        df = fetch_prices(ticker, period="5d", interval="5m")
        if df is None or df.empty:
            return None, None, False
        c = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        last_ts = c.index[-1]
        if last_ts.tz is None:
            last_ts = last_ts.tz_localize("UTC")
        active = (datetime.now(timezone.utc) - last_ts) <= timedelta(minutes=20)
        drift_60 = float(c.iloc[-1] / c.iloc[-13] - 1.0) if len(c) >= 13 else None
        today_mask = (c.index.date == c.index[-1].date())
        drift_sess = float(c.iloc[-1] / c.loc[today_mask].iloc[0] - 1.0) if today_mask.any() else None
        return drift_60, drift_sess, bool(active)
    except Exception:
        return None, None, False


# ---------------- Probability blend + sizing ----------------
def compute_probability_and_reasons(
    ticker: str,
    capital: float = 10000.0,
    risk_pct: float = 1.0,
    atr_mult: float = 1.5,
    rr_target: float = 1.5,
) -> Tuple[float, Dict[str, Any]]:
    df = fetch_prices(ticker, period="6mo", interval="1d")
    if df is None or df.empty:
        return 0.5, {"error": f"No data for {ticker}"}

    p = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    o, h, l = df["Open"], df["High"], df["Low"]

    ema20, ema50 = ema(p, 20), ema(p, 50)
    sma200 = p.rolling(200).mean()
    _, upBB, loBB = bollinger(p, 20, 2)
    _, _, hist = macd(p)
    rsi_s = rsi(p, 14)

    # Tech edge (bounded)
    px = float(p.iloc[-1])
    e20 = float(ema20.iloc[-1]) if pd.notna(ema20.iloc[-1]) else px
    e50 = float(ema50.iloc[-1]) if pd.notna(ema50.iloc[-1]) else px
    rsi_v = float(rsi_s.iloc[-1]) if pd.notna(rsi_s.iloc[-1]) else 50.0
    hist_v = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else 0.0

    ema_rel = np.tanh(((e20 - e50) / max(1e-6, abs(e50))) * 6)
    rsi_rel = np.tanh((rsi_v - 50) / 10)
    macd_rel = np.tanh(hist_v * 5)
    tech_edge = 0.55*ema_rel + 0.30*rsi_rel + 0.15*macd_rel

    # News edge (recency-weighted)
    news_df = get_news_with_sentiment(ticker)
    news_edge = float(np.tanh(news_df["weighted_sent"].sum())) if (isinstance(news_df, pd.DataFrame) and not news_df.empty) else 0.0

    # Macro edge
    macro = fetch_macro(); vix = macro.get("vix", pd.DataFrame()); spy = macro.get("spy", pd.DataFrame())
    macro_edge = 0.0
    if not vix.empty and "Close" in vix.columns:
        vclose = vix["Close"].astype(float).dropna()
        if len(vclose):
            vix_pct = float(np.asarray(vclose.rank(pct=True).iloc[-1]).ravel()[0])
            macro_edge += -0.8*max(0.0, vix_pct-0.5)
    if not spy.empty and "Close" in spy.columns and spy["Close"].shape[0] > 50:
        sc = spy["Close"].astype(float).dropna()
        se50 = ema(sc, 50)
        s_trend = float(np.tanh(((float(sc.iloc[-1])-float(se50.iloc[-1]))/max(1e-6, float(se50.iloc[-1]))) * 10))
        macro_edge += 0.5*s_trend

    # Intraday tie-breaker
    drift60, drifts, active = intraday_drift_5m(ticker)
    intra_edge = 0.0
    if active and (drift60 is not None):
        intra_edge += 0.35 * float(np.tanh(drift60 * 20))
    if active and (drifts is not None):
        intra_edge += 0.20 * float(np.tanh(drifts * 10))

    # Blend -> probability
    z = 1.25*news_edge + 0.90*tech_edge + 0.50*macro_edge + intra_edge
    prob_up = 1.0 / (1.0 + math.exp(-z))

    # Position sizing (ATR-based)
    atr_s = atr(h, l, p, 14)
    atr_now = float(atr_s.iloc[-1]) if pd.notna(atr_s.iloc[-1]) else 0.0
    stop_dist = float(atr_mult) * atr_now
    risk_amt = max(1.0, float(capital) * (float(risk_pct)/100.0))
    shares = int(max(0, math.floor(risk_amt / stop_dist))) if stop_dist > 0 else 0
    take_profit = float(rr_target) * stop_dist

    meta = {
        "ticker": ticker,
        "prob_pct": round(prob_up*100, 1),
        "price": round(float(px), 2),
        "drift60": drift60,
        "drifts": drifts,
        "active": active,
        "atr": atr_now,
        "stop_dist": stop_dist,
        "take_profit": take_profit,
        "risk_amt": risk_amt,
        "shares": shares,
    }
    series = {
        "p": p, "o": o, "h": h, "l": l,
        "ema20": ema20, "ema50": ema50, "sma200": sma200,
        "upBB": upBB, "loBB": loBB, "rsi": rsi_s
    }
    return prob_up, {"meta": meta, "series": series, "news": news_df}


# ---------------- Chart builders ----------------
def price_with_rsi_chart(series: Dict[str, pd.Series], ticker: str, price: float) -> go.Figure:
    if not series or "p" not in series or series["p"] is None or series["p"].empty:
        return empty_figure(title=f"{ticker}: no price data", height=640)

    p, o, h, l = series["p"], series["o"], series["h"], series["l"]
    ema20, ema50, sma200 = series["ema20"], series["ema50"], series["sma200"]
    upBB, loBB, rsi_s = series["upBB"], series["loBB"], series["rsi"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{ticker} — Price & Signals", "RSI(14)")
    )
    # Price panel
    fig.add_trace(go.Candlestick(x=p.index, open=o, high=h, low=l, close=p, name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=ema20, name="EMA20", line=dict(width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=ema50, name="EMA50", line=dict(width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=sma200, name="SMA200", line=dict(dash="dash", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=upBB, name="Upper BB", line=dict(dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=loBB, name="Lower BB", line=dict(dash="dot", width=1)), row=1, col=1)

    # Current price
    fig.add_hline(y=price, line=dict(color="#666", width=1, dash="dot"), row=1, col=1)
    fig.add_annotation(x=p.index[-1], y=price, xanchor="right", yanchor="bottom",
                       text=f" ${price:,.2f}", showarrow=False, font=dict(size=14))

    # RSI panel
    fig.add_trace(go.Scatter(x=p.index, y=rsi_s, name="RSI14", line=dict(width=1.8)), row=2, col=1)
    fig.add_hline(y=70, line=dict(dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(dash="dash", width=1), row=2, col=1)

    fig.update_layout(
        title_font_size=22,
        xaxis_rangeslider_visible=False,
        autosize=False,
        height=640,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def probability_gauge(ticker: str, pct: float, mini: bool = False) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": f"{ticker} — Chance Up", "font": {"size": 20 if not mini else 16}},
        number={"suffix": "%", "font": {"size": 36 if not mini else 24}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.25},
            "steps": [
                {"range": [0, 40], "color": "#ffe6e6"},
                {"range": [40, 60], "color": "#fff3cc"},
                {"range": [60, 100], "color": "#e6f7e6"}
            ]
        }
    ))
    fig.update_layout(autosize=False, height=320 if not mini else 210, margin=dict(l=10, r=10, t=40, b=6))
    return fig

# app.py
# -*- coding: utf-8 -*-
"""
Market Decision Dashboard — modular (imports from marketdash.utils)
Run:
  python app.py  ->  http://127.0.0.1:8080
"""

from typing import List
import pandas as pd
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from marketdash.utils import (
    compute_probability_and_reasons,
    price_with_rsi_chart,
    probability_gauge,
)

DEFAULT_TICKERS: List[str] = ["HOOD", "NVDA", "AMD", "META", "AMAT", "TSLA", "AMZA", "MSFT"]
ALIAS = {"NVDIA": "NVDA"}  # common typo

external_stylesheets = [dbc.themes.FLATLY]
app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# ---- Controls
controls = dbc.Card([
    dbc.CardHeader("Controls"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Detail ticker", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="detail-ticker",
                    options=[{"label": t, "value": t} for t in DEFAULT_TICKERS],
                    value=DEFAULT_TICKERS[0],
                    clearable=False
                )
            ], xs=12, sm=6, md=4, lg=3),
            dbc.Col([
                html.Label("Tiles (watchlist)", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="tile-tickers",
                    options=[{"label": t, "value": t} for t in DEFAULT_TICKERS],
                    value=DEFAULT_TICKERS,
                    multi=True
                )
            ], xs=12, sm=12, md=8, lg=6),
            dbc.Col([
                html.Button("Refresh", id="refresh", className="btn btn-primary", style={"marginTop": "28px"})
            ], xs=6, sm=6, md=4, lg=3),
        ], className="g-2"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Label("Account size ($)", style={"fontWeight": 600}),
                dcc.Input(id="capital", type="number", value=10000, min=1000, step=500, style={"width": "100%"})
            ], xs=6, sm=6, md=3),
            dbc.Col([
                html.Label("Risk / trade (%)", style={"fontWeight": 600}),
                dcc.Input(id="risk_pct", type="number", value=1.0, min=0.1, step=0.1, style={"width": "100%"})
            ], xs=6, sm=6, md=3),
            dbc.Col([
                html.Label("ATR stop (×)", style={"fontWeight": 600}),
                dcc.Input(id="atr_mult", type="number", value=1.5, min=0.5, step=0.1, style={"width": "100%"})
            ], xs=6, sm=6, md=3),
            dbc.Col([
                html.Label("RR target (R)", style={"fontWeight": 600}),
                dcc.Input(id="rr_target", type="number", value=1.5, min=0.5, step=0.5, style={"width": "100%"})
            ], xs=6, sm=6, md=3),
        ], className="g-2"),
    ])
], className="mb-3")

# ---- Layout
def _graph(id_, height_px):
    return dcc.Loading(
        dcc.Graph(id=id_, className="graph-100", style={"height": f"{height_px}px"}, config={"responsive": True}),
        type="dot"
    )

app.layout = dbc.Container([
    html.H2("📈 Market Decision Dashboard — Responsive", style={"fontSize": "28px", "fontWeight": 700}),
    controls,
    dbc.Row([
        dbc.Col(_graph("detail-chart", 640), lg=8, md=12),
        dbc.Col([
            _graph("detail-gauge", 320),
            html.Div(id="news-tape", className="mt-2"),
            dbc.Card([
                dbc.CardHeader("Position Sizing (ATR)"),
                dbc.CardBody(id="sizing-box")
            ], className="mt-2")
        ], lg=4, md=12)
    ], className="g-3"),
    html.Hr(),
    html.H4("🧩 Multi-Tile Probability Dashboard", style={"fontWeight": 700}),
    dbc.Row(id="tiles-row", className="g-3"),
], fluid=True)

# ---- Callbacks
@app.callback(
    [Output("detail-chart", "figure"),
     Output("detail-gauge", "figure"),
     Output("news-tape", "children"),
     Output("sizing-box", "children")],
    [Input("detail-ticker", "value"), Input("refresh", "n_clicks")],
    [State("capital", "value"), State("risk_pct", "value"), State("atr_mult", "value"), State("rr_target", "value")]
)
def update_detail(ticker, _n, capital, risk_pct, atr_mult, rr_target):
    t = (ALIAS.get(ticker, ticker) if ticker else DEFAULT_TICKERS[0])
    prob, out = compute_probability_and_reasons(
        t,
        capital=float(capital or 10000),
        risk_pct=float(risk_pct or 1.0),
        atr_mult=float(atr_mult or 1.5),
        rr_target=float(rr_target or 1.5),
    )
    if "error" in out:
        # placeholders
        from marketdash.utils import empty_figure, probability_gauge
        empty = empty_figure(f"{t}: data not available", height=640)
        gauge = probability_gauge(t, 50.0, mini=False)
        return empty, gauge, html.Div("(no recent headlines)"), [html.Div("ATR(14): n/a")]

    meta, series, news = out.get("meta", {}), out.get("series", {}), out.get("news", pd.DataFrame())
    fig_chart = price_with_rsi_chart(series, meta.get("ticker", t), float(meta.get("price", 0.0) or series["p"].iloc[-1]))
    fig_gauge = probability_gauge(meta.get("ticker", t), meta.get("prob_pct", 50.0), mini=False)

    # News tape
    items = []
    if isinstance(news, pd.DataFrame) and not news.empty:
        for _, row in news.head(5).iterrows():
            sent = float(row.get("sentiment", 0.0))
            col = "#228B22" if sent > 0.2 else ("#B22222" if sent < -0.2 else "#555")
            ts = row.get("published")
            try:
                ts_txt = pd.to_datetime(ts).strftime("%b %d %H:%M") if pd.notna(ts) else ""
            except Exception:
                ts_txt = ""
            items.append(html.Li([
                html.Span(ts_txt + " — "),
                html.Span(str(row.get("title", "")), style={"color": col})
            ], style={"fontSize": "0.92rem", "marginBottom": "4px"}))
    news_children = [html.Strong("News (last 5)"), html.Ul(items)] if items else html.Div("(no recent headlines)")

    # Sizing box
    drift_bits = []
    if meta.get("active"):
        if meta.get("drift60") is not None:
            drift_bits.append(f"60m drift: {meta['drift60']*100:+.2f}%")
        if meta.get("drifts") is not None:
            drift_bits.append(f"Session drift: {meta['drifts']*100:+.2f}%")
    sizing_children = [
        html.Div(f"ATR(14): {meta.get('atr', 0.0):.2f}"),
        html.Div(f"Stop: {float(atr_mult or 1.5):.1f}×ATR → ${meta.get('stop_dist', 0.0):.2f}"),
        html.Div(f"Take-profit: {float(rr_target or 1.5):.1f}R → ${meta.get('take_profit', 0.0):.2f}"),
        html.Div(f"Risk per trade: ${meta.get('risk_amt', 0.0):.2f}"),
        html.Div(f"Suggested size: {int(meta.get('shares', 0))} sh @ ${meta.get('price', 0.0):.2f}"),
    ]
    if drift_bits:
        sizing_children.append(html.Div(" | ".join(drift_bits), style={"color": "#555", "fontSize": "0.9rem"}))

    return fig_chart, fig_gauge, news_children, sizing_children

@app.callback(
    Output("tiles-row", "children"),
    [Input("tile-tickers", "value"), Input("refresh", "n_clicks")]
)
def update_tiles(tickers, _n):
    from marketdash.utils import probability_gauge
    if not tickers:
        tickers = DEFAULT_TICKERS
    tiles = []
    for t in tickers[:12]:
        try:
            _, out = compute_probability_and_reasons(ALIAS.get(t, t))
            meta = out.get("meta", {})
            fig = probability_gauge(meta.get("ticker", t), meta.get("prob_pct", 50.0), mini=True)
        except Exception:
            fig = probability_gauge(t, 50.0, mini=True)
        card = dbc.Card(dcc.Graph(figure=fig, className="graph-100 tile-graph", config={"responsive": True}),
                        className="tile-card h-100")
        tiles.append(dbc.Col(card, xs=12, sm=6, md=4, lg=3))
    return tiles

if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port=8080, debug=False)


# assets/app.css
"""assets/app.css
/* Fill columns */
.graph-100 { width: 100%; }

/* Force a stable height for tile graphs & cards */
.tile-card { min-height: 230px; }
.tile-graph { height: 210px !important; }

@media (max-width: 1200px) {
  .tile-card { min-height: 220px; }
  .tile-graph { height: 200px !important; }
}
@media (max-width: 992px) { /* laptops */
  .tile-card { min-height: 210px; }
  .tile-graph { height: 190px !important; }
}
@media (max-width: 768px) { /* tablets/phones */
  .tile-card { min-height: 200px; }
  .tile-graph { height: 180px !important; }
}
"""



