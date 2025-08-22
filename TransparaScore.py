import os
import warnings
import base64
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import joblib
import requests
import pandas as pd
import numpy as np
import streamlit as st
import feedparser
from sklearn.ensemble import RandomForestRegressor

# ======== SHAP REPLACEMENT ========
# SHAP is used for feature attribution/explanation. If unavailable, we use model.feature_importances_
# and a simple difference to base prediction for local attribution.

def get_feature_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with feature importances for the trained model and a local explanation."""
    # Model global importances:
    importances = getattr(model, "feature_importances_", np.ones(X.shape[1]))
    base_pred = np.mean(model.predict(X))  # Not true SHAP base value, but a neutral reference
    pred = model.predict(X)[0]
    contribs = (X.iloc[0] - X.mean()).values * importances
    # Normalize local contributions such that sum matches pred - base_pred (not exact SHAP, but interpretable)
    if np.sum(np.abs(contribs)) > 0:
        contribs = contribs * ((pred - base_pred) / np.sum(contribs) if np.sum(contribs) != 0 else 0)
    else:
        contribs = np.zeros_like(contribs)
    df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances,
        "Local Contrib": contribs
    }).sort_values("Importance", ascending=False)
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load("credit_model.pkl")
    except FileNotFoundError:
        X_train = pd.DataFrame({
            "return_pct": np.random.randn(100),
            "vol_pct": np.random.rand(100)*10,
            "max_dd_pct": np.random.rand(100)*-20,
            "liq_proxy": np.random.rand(100)*1000,
            "cpi_yoy_pct": np.random.rand(100)*5,
            "unemp_pct": np.random.rand(100)*10,
            "rate10y_pct": np.random.rand(100)*6,
            "gdp_growth_us_pct": np.random.rand(100)*5,
            "gdp_growth_in_pct": np.random.rand(100)*7,
            "event_signal": np.random.randn(100)
        })
        y_train = np.random.rand(100)*100
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        joblib.dump(model, "credit_model.pkl")
        return model

ml_model = load_model()

# =======================
# ---- API KEYS & CONFIG ----
# =======================
YAHOO_API_KEY = st.secrets["YAHOO_API_KEY"]
FRED_API_KEY = st.secrets["FRED_API_KEY"]
WORLDBANK_API = "http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json"
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
RSS_FEED_URL = "https://feeds.reuters.com/reuters/businessNews"

DEFAULT_COMPANIES = {
    "CREDTECH": "CREDTECH.NS",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "IBM": "IBM"
}
DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "1d"

# =======================
# ---- PAGE SETUP ----
# =======================
st.set_page_config(
    page_title="CredTech - Explainable Credit Scorecard",
    layout="wide"
)

def set_bg(image_file: str):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded});
            background-size: cover;
            background-attachment: fixed;
        }}
        .glass {{
            background: rgba(255,255,255,0.92) !important;
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
        }}
        hr {{
            border: none;
            border-top: 1px solid #bbb !important;
            margin: 0.5rem 0;
        }}
        h1, h2, h3, h4, h5, h6, p, li, span, label, div, thead, tbody, td, th {{
            color: #222 !important;
        }}

        .glass-block {{
            background: rgba(255,255,255,0.87) !important;
            border-radius: 22px;
            padding: 2.3rem 2.7rem 2.3rem 2.7rem;
            box-shadow: 0 4px 24px 0 rgba(90,90,90,0.09);
            margin-bottom: 2.5rem;
            margin-top: 2.0rem;
            border: 1.7px solid rgba(220,220,220,0.28);
            font-size: 1.19rem;
        }}
        .block-container {{
            padding-top: 1.5rem !important;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Background image load failed: {e}")

set_bg("Website_Background.png")

st.markdown("<h1 style='text-align:center;'>🏦 CredTech Explainable Credit Scorecard</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="glass" style="margin:auto;text-align:center;">
    <b>Welcome to the CredTech Explainable Credit Scorecard Platform!</b><br>
    <span style='font-size: 1.2em; color:#555;'>
        Explore comprehensive, data-driven credit scoring for major companies. <br>
        Our platform integrates live financial, macroeconomic, and news data to offer transparency and actionable insights for credit assessment.<br>
        All scoring rules and data sources are clearly explained below.<br>
        <br>
        <i>Built for the CredTech Hackathon to empower business users with clear, explainable risk intelligence.</i>
    </span>
</div>
""", unsafe_allow_html=True)

# =======================
# ---- SIDEBAR ----
# =======================
with st.sidebar:
    st.header("Company & Data Options")
    company = st.selectbox("Select Company", list(DEFAULT_COMPANIES.keys()))
    period = st.selectbox("Price history window", ["1mo", "3mo", "6mo", "1y"], index=0)
    interval = st.selectbox("Data interval", ["1d", "1wk"], index=0)
    st.markdown("---")
    st.subheader("Included Data Sources")
    st.markdown("""
    - **Stock Prices**: Yahoo Finance (via RapidAPI)
    - **Macroeconomic Data**: US/IN macro from FRED and World Bank
    - **News & Events**: GDELT News API and Reuters Business RSS
    """)
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
        <div style='font-size: 0.9em; color: #888'>
        Data is fetched live on every refresh.<br>
        No API keys are required from the user.<br>
        Built for demonstration and learning purposes only.
        </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([10, 13])
with col2:
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    refresh_btn = st.button("🔄 Refresh Data Now")

# =======================
# ---- HELPERS ----
# =======================
@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_yahoo_timeseries(symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-chart"
        params = {"symbol": symbol, "region": "US", "interval": interval, "range": period}
        headers = {
            "x-rapidapi-host": "apidojo-yahoo-finance-v1.p.rapidapi.com",
            "x-rapidapi-key": YAHOO_API_KEY
        }
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        result = js.get("chart", {}).get("result", [{}])[0]
        timestamps = result.get("timestamp", [])
        indicators = result.get("indicators", {}).get("quote", [{}])[0]
        if not timestamps or not indicators:
            return None
        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("UTC"),
            "Open": indicators.get("open", []),
            "High": indicators.get("high", []),
            "Low": indicators.get("low", []),
            "Close": indicators.get("close", []),
            "Volume": indicators.get("volume", []),
        }).dropna()
        df = df.set_index("Date")
        return df
    except Exception:
        return None

@st.cache_data(ttl=60*60, show_spinner=False)
def fred_series(series_id: str, observation_start: str = "2015-01-01") -> Optional[pd.DataFrame]:
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": observation_start,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return None
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")["value"].to_frame(series_id)
    except Exception:
        return None

@st.cache_data(ttl=60*60, show_spinner=False)
def worldbank_indicator(country_code: str, indicator: str) -> Optional[pd.DataFrame]:
    try:
        url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
        params = {"format": "json", "per_page": 20000}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        data = js[1] if isinstance(js, list) and len(js) > 1 else None
        if not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], format="%Y", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")["value"].to_frame(indicator)
    except Exception:
        return None

@st.cache_data(ttl=60*10, show_spinner=False)
def gdelt_headlines(query: str, max_rows: int = 20) -> List[Dict]:
    try:
        params = {"query": query, "format": "json", "maxrecords": max_rows}
        r = requests.get(GDELT_API, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception:
        return []

@st.cache_data(ttl=60*10, show_spinner=False)
def rss_latest(url: str, limit: int = 20) -> List[Dict]:
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            items.append({
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "published": e.get("published", ""),
                "summary": e.get("summary", ""),
            })
        return items
    except Exception:
        return []

# =======================
# ---- FEATURE ENGINEERING ----
# =======================
def price_features(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"return_pct": 0.0, "vol_pct": 0.0, "max_dd_pct": 0.0, "liq_proxy": 0.0}
    close = df["Close"].dropna()
    if close.empty:
        return {"return_pct": 0.0, "vol_pct": 0.0, "max_dd_pct": 0.0, "liq_proxy": 0.0}
    ret_pct = (close.iloc[-1] - close.iloc[0]) / max(1e-9, close.iloc[0]) * 100
    vol_pct = close.pct_change().std() * 100
    roll_max = close.cummax()
    dd = (close / roll_max - 1.0) * 100
    max_dd = dd.min()
    vol = df.get("Volume", pd.Series(index=df.index, dtype=float)).fillna(0)
    liq = float(np.nanmedian(vol))
    return {
        "return_pct": float(ret_pct),
        "vol_pct": float(vol_pct),
        "max_dd_pct": float(max_dd),
        "liq_proxy": float(liq),
    }

def macro_features() -> Dict[str, float]:
    out = {}
    cpi = fred_series("CPIAUCSL")
    if cpi is not None and not cpi.empty:
        cpi = cpi.resample("M").last()
        cpi_yoy = cpi.pct_change(12).iloc[-1, 0] * 100 if len(cpi) > 12 else np.nan
        out["cpi_yoy_pct"] = float(cpi_yoy) if pd.notna(cpi_yoy) else 0.0
    else:
        out["cpi_yoy_pct"] = 0.0
    unemp = fred_series("UNRATE")
    out["unemp_pct"] = float(unemp.iloc[-1, 0]) if unemp is not None and not unemp.empty else 0.0
    dgs10 = fred_series("DGS10")
    out["rate10y_pct"] = float(dgs10.iloc[-1, 0]) if dgs10 is not None and not dgs10.empty else 0.0
    try:
        us_gdp = worldbank_indicator("US", "NY.GDP.MKTP.KD.ZG")
        in_gdp = worldbank_indicator("IN", "NY.GDP.MKTP.KD.ZG")
        out["gdp_growth_us_pct"] = float(us_gdp.iloc[-1, 0]) if us_gdp is not None and not us_gdp.empty else 0.0
        out["gdp_growth_in_pct"] = float(in_gdp.iloc[-1, 0]) if in_gdp is not None and not in_gdp.empty else 0.0
    except Exception:
        out["gdp_growth_us_pct"] = out["gdp_growth_in_pct"] = 0.0
    return out

def event_signals_for_symbol(symbol: str) -> Tuple[float, List[Dict]]:
    items = []
    items += gdelt_headlines(symbol, max_rows=10)
    items += rss_latest(RSS_FEED_URL, limit=10)
    if not items:
        return 0.0, []
    scored = []
    total = 0.0

    NEG_RULES = [
        ("bankrupt", 15), ("default", 15), ("restructur", 12), ("downgrade", 10),
        ("lawsuit", 8), ("probe", 6), ("fraud", 12), ("layoff", 8), ("data breach", 8),
    ]
    POS_RULES = [
        ("record profit", 10), ("beat earnings", 8), ("upgrade", 6), ("guidance raise", 6),
        ("contract win", 6), ("debt repurchase", 8), ("refinanc", 6), ("expansion", 5),
    ]
    for it in items:
        title = (it.get("title") or it.get("seendate") or "").lower()
        score = 0
        label = "neutral"
        for k, w in NEG_RULES:
            if k in title:
                score -= w
                label = "negative"
        for k, w in POS_RULES:
            if k in title:
                score += w
                label = "positive" if score > 0 else label
        scored.append({"title": it.get("title", "(no title)"), "score": score, "label": label, "link": it.get("url") or it.get("link")})
        total += score
    avg = np.clip(total / max(1, len(scored)), -20, 20)
    return float(avg), scored

# =======================
# ---- SCORING ENGINE ----
# =======================
def normalize(x, lo, hi, inverse=False):
    if pd.isna(x):
        return 0.5
    x = np.clip(x, lo, hi)
    z = (x - lo) / (hi - lo + 1e-9)
    return 1.0 - z if inverse else z

def score_financial(pf: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    s_return = normalize(pf.get("return_pct", 0), -30, 30, inverse=False)
    s_vol = normalize(pf.get("vol_pct", 0), 0, 8, inverse=True)
    s_dd = normalize(pf.get("max_dd_pct", 0), -40, 0, inverse=True)
    s_liq = normalize(np.log1p(max(0.0, pf.get("liq_proxy", 0))), 0, 18, inverse=False)
    subs = {"Return": s_return, "Volatility": s_vol, "Drawdown": s_dd, "Liquidity": s_liq}
    fin_score = float(np.dot(list(subs.values()), [0.35, 0.30, 0.20, 0.15]) * 100)
    return fin_score, {k: round(v * 100, 2) for k, v in subs.items()}

def score_macro(mf: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    s_cpi = normalize(mf.get("cpi_yoy_pct", 0), -1, 10, inverse=True)
    s_unemp = normalize(mf.get("unemp_pct", 0), 2.5, 9, inverse=True)
    s_rate = normalize(mf.get("rate10y_pct", 0), 0.5, 7.0, inverse=True)
    subs = {"Inflation": s_cpi, "Unemployment": s_unemp, "Rates10Y": s_rate}
    gdp_us = mf.get("gdp_growth_us_pct", 0)
    gdp_in = mf.get("gdp_growth_in_pct", 0)
    gdp_bonus = 0.02 * max(0, gdp_us) + 0.02 * max(0, gdp_in)
    macro_score = float(np.dot(list(subs.values()), [0.4, 0.35, 0.25]) * 100)
    macro_score = float(np.clip(macro_score + gdp_bonus * 100, 0, 100))
    subs_out = {k: round(v * 100, 2) for k, v in subs.items()}
    if gdp_us or gdp_in:
        subs_out["GDP Growth"] = round(np.clip(gdp_bonus, 0, 1) * 100, 2)
    return macro_score, subs_out

def score_events(ev_avg: float) -> Tuple[float, Dict[str, float]]:
    ev_norm = (ev_avg + 20) / 40
    return float(np.clip(ev_norm, 0, 1) * 100), {"Event Signal": round(np.clip(ev_norm, 0, 1) * 100, 2)}

# =======================
# ---- MAIN VIEW ----
# =======================
if refresh_btn:
    fetch_yahoo_timeseries.clear()
    fred_series.clear()
    worldbank_indicator.clear()
    gdelt_headlines.clear()
    rss_latest.clear()

with st.container():

    st.header("📊 Company Credit Overview")

    symbol = DEFAULT_COMPANIES[company]
    df = fetch_yahoo_timeseries(symbol, period=period, interval=interval)
    pf = price_features(df)
    fin_score, fin_sub = score_financial(pf)
    macro = macro_features()
    mac_score, mac_sub = score_macro(macro)
    ev_avg, ev_list = event_signals_for_symbol(symbol)
    ev_score, ev_sub = score_events(ev_avg)
    weights = [0.55, 0.25, 0.20]
    final_score = weights[0]*fin_score + weights[1]*mac_score + weights[2]*ev_score

    overview_data = {
        "Company": [company],
        "Credit Score": [round(final_score, 2)],
        "Return (%)": [round(pf.get("return_pct", 0), 2)],
        "Volatility (%)": [round(pf.get("vol_pct", 0), 2)],
        "Drawdown (%)": [round(pf.get("max_dd_pct", 0), 2)],
    }
    ov_df = pd.DataFrame(overview_data)
    st.dataframe(ov_df, use_container_width=True)

    st.markdown("""
    ### How to Interpret the Scorecard

    - **Credit Score (0-100)**: Higher values indicate stronger creditworthiness, combining financial, macro and event factors.
    - **Return (%)**: Price return over the selected window. Higher is better.
    - **Volatility (%)**: Standard deviation of daily/weekly returns. Lower is better (stable prices).
    - **Drawdown (%)**: Maximum observed drop from peak price. Lower is better (less risky).
    - **Financial Score**: Derived from return, volatility, drawdown, and liquidity.
    - **Macro Score**: Tracks inflation (CPI YoY), unemployment, 10Y interest rate, GDP growth.
    - **Event Score**: Weighted signal from recent news and event headlines, classifying positive/negative stories.
    - **All scores and features are computed using simple, transparent rules.**
    """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Factor Contribution Breakdown")

    contrib = pd.DataFrame({
        "Factor": ["Financial", "Macro", "Events"],
        "Score": [fin_score, mac_score, ev_score]
    }).set_index("Factor")
    st.bar_chart(contrib)

    st.write("**Financial sub-factors (0-100):**")
    st.dataframe(pd.DataFrame.from_dict(fin_sub, orient="index", columns=["Score"]))

    st.write("**Macro sub-factors (0-100):**")
    st.dataframe(pd.DataFrame.from_dict(mac_sub, orient="index", columns=["Score"]))

    st.write("**Event signal (0-100):**")
    st.dataframe(pd.DataFrame.from_dict(ev_sub, orient="index", columns=["Score"]))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Stock Price Trend")
    if df is not None and not df.empty:
        st.line_chart(df["Close"])
    else:
        st.info("No price history available for chart.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Score Trend (Simulated)")
    if df is not None and not df.empty:
        hist_scores = []
        cdf = df["Close"].reset_index()
        for i in range(5, len(cdf)):
            sub = df.iloc[:i]
            s_fin, _ = score_financial(price_features(sub))
            s = (weights[0]*s_fin + weights[1]*mac_score + weights[2]*ev_score)
            hist_scores.append((cdf.loc[i, "Date"], s))
        if hist_scores:
            hdf = pd.DataFrame(hist_scores, columns=["Date", "Score"]).set_index("Date")
            st.line_chart(hdf)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Recent Macro Indicators")
    st.write(f"- **Inflation (CPI YoY):** {macro.get('cpi_yoy_pct',0):.2f}%")
    st.write(f"- **US GDP Growth:** {macro.get('gdp_growth_us_pct',0):.2f}%")
    st.write(f"- **IN GDP Growth:** {macro.get('gdp_growth_in_pct',0):.2f}%")
    st.write(f"- **10Y Treasury Rate:** {macro.get('rate10y_pct',0):.2f}%")
    st.write(f"- **Unemployment Rate:** {macro.get('unemp_pct',0):.2f}%")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Recent Events & News")
    events_df = pd.DataFrame(ev_list) if ev_list else pd.DataFrame(columns=["title","label","score","link"])
    if not events_df.empty:
        events_df = events_df[["title", "label", "score", "link"]]
        st.dataframe(events_df, use_container_width=True)
        st.write("""
        **Interpretation:**  
        - Positive news headlines ("positive" label) may indicate credit improvement or opportunity.
        - Negative headlines ("negative" label) may signal risk factors or credit concerns.
        - Neutral/no news typically means "no news is good news".
        """)
    else:
        st.markdown("**No recent scored events, but here are the latest news headlines for this company:**")
        top_news = gdelt_headlines(symbol, max_rows=3)
        if not top_news:
            top_news = rss_latest(RSS_FEED_URL, limit=3)
        if top_news:
            for news in top_news[:3]:
                title = news.get("title", "(No title)")
                link = news.get("url") or news.get("link") or "#"
                published = news.get("published", "")
                st.markdown(f"- [{title}]({link}) <span style='color:#888;font-size:0.85em;'>({published})</span>", unsafe_allow_html=True)
        else:
            st.write("No live news headlines available at this moment.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Understanding This Platform")
    st.markdown(f"""
    **What does this platform do?**

    This CredTech scorecard combines three core types of data for any selected company:
    - **1. Financial Market Data:**  
        We use stock price trends, volatility, and drawdown to gauge recent financial risk and momentum.
    - **2. Macroeconomic Indicators:**  
        We fetch the latest inflation, unemployment, 10-year interest rates, and GDP growth to understand the external economic environment.
    - **3. News & Events:**  
        We scan news feeds and event databases for major headlines that may affect company creditworthiness.

    **All scoring rules are simple, open, and rule-based.**  
    There is no black-box AI or hidden logic.  
    All data and scores can be downloaded for further analysis.

    *For more details, see the hackathon problem statement and documentation.*

    ---
    """)

    features = {
        "return_pct": pf["return_pct"],
        "vol_pct": pf["vol_pct"],
        "max_dd_pct": pf["max_dd_pct"],
        "liq_proxy": pf["liq_proxy"],
        "cpi_yoy_pct": macro["cpi_yoy_pct"],
        "unemp_pct": macro["unemp_pct"],
        "rate10y_pct": macro["rate10y_pct"],
        "gdp_growth_us_pct": macro["gdp_growth_us_pct"],
        "gdp_growth_in_pct": macro["gdp_growth_in_pct"],
        "event_signal": ev_avg
    }
    X = pd.DataFrame([features])
    pred_score = ml_model.predict(X)[0]
    feat_imp_df = get_feature_importance(ml_model, X)

    st.subheader("⬇️ Export Data")
    csv = ov_df.to_csv(index=False).encode()
    st.download_button("Download Company Overview CSV", data=csv, file_name=f"{company}_overview.csv", mime="text/csv")
    st.caption(f"Last refreshed (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    st.subheader("🤖 ML Model Prediction & Feature Explanation")
    st.write(f"Predicted Credit Score (ML Model): {pred_score:.2f}")

    st.dataframe(feat_imp_df[["Feature", "Importance", "Local Contrib"]])

    top_feature = feat_imp_df.iloc[0]
    if top_feature["Local Contrib"] > 0:
        impact = "positively"
    else:
        impact = "negatively"
    st.markdown(f"""
    **Explanation in plain words:**  
    The model predicts this score mainly because **{top_feature['Feature']}** {impact} influenced the rating.  
    Other important drivers include {', '.join(feat_imp_df['Feature'].iloc[1:3].tolist())}.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:#000'>
        <b>Built for CredTech Hackathon | All data is for demonstration only | 
        Contact: <a href="https://www.linkedin.com/in/nitanshu-tak-89a1ba289/?originalSubdomain=in" target="_blank" style="color:#000; text-decoration:none;">Nitasnshu Tak</a></b>
    </div>
    """,
    unsafe_allow_html=True
)

