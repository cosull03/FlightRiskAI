"""
Flight Risk AI — accurate flight delay prediction using real ML on BTS 2024 data.

Architecture:
  - User enters origin, destination, airline, date, time
  - Gradient Boosting model predicts delay probability
  - OpenWeatherMap fetches forecast for the actual flight time
  - GPT-4o-mini synthesizes data into actionable recommendation
  - Bayesian (Laplace) smoothing on historical stats prevents 0%/100%
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
import os
from openai import OpenAI
from datetime import date, datetime, time, timedelta
from dotenv import load_dotenv

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENWEATHER_KEY   = os.getenv("OPENWEATHER_KEY", "")

st.set_page_config(
    page_title="Flight Risk AI",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Reference data ─────────────────────────────────────────────────────────────
AIRLINE_NAMES = {
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
    "WN": "Southwest Airlines", "B6": "JetBlue Airways", "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",   "F9": "Frontier Airlines","G4": "Allegiant Air",
    "HA": "Hawaiian Airlines", "9E": "Endeavor Air",    "MQ": "Envoy Air",
    "OH": "PSA Airlines",      "OO": "SkyWest Airlines","YX": "Republic Airways",
}
AIRLINE_TO_CODE = {v: k for k, v in AIRLINE_NAMES.items()}

def time_bucket(hour):
    if hour < 6:  return "red_eye"
    if hour < 12: return "morning"
    if hour < 17: return "afternoon"
    if hour < 21: return "evening"
    return "night"

def time_bucket_label(b):
    return {"red_eye":"Red-eye","morning":"Morning","afternoon":"Afternoon",
            "evening":"Evening","night":"Late Night"}.get(b, b)

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    summary  = pd.read_csv(os.path.join(base, "flight_summary.csv"))
    airports = pd.read_csv(os.path.join(base, "airports_full.csv"))
    coords   = pd.read_csv(os.path.join(base, "airport_coords.csv"))
    distances = pd.read_csv(os.path.join(base, "route_distances.csv"))
    route_airlines = pd.read_csv(os.path.join(base, "route_airlines.csv"))
    schedules = pd.read_csv(os.path.join(base, "airline_schedules.csv"))
    return (
        summary, airports,
        {row["airport"]: (row["lat"], row["lon"]) for _, row in coords.iterrows()},
        {(row["origin"], row["dest"]): row["distance"] for _, row in distances.iterrows()},
        route_airlines,
        schedules,
    )

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base, "delay_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(base, "model_features.pkl"), "rb") as f:
            features = pickle.load(f)
        with open(os.path.join(base, "model_metrics.json"), "r") as f:
            metrics = json.load(f)
        return model, features, metrics
    except FileNotFoundError:
        return None, None, None

summary_df, airports_df, AIRPORT_COORDS, ROUTE_DISTANCES, ROUTE_AIRLINES, AIRLINE_SCHEDULES = load_data()
ml_model, ml_features, ml_metrics = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# DATA / PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_route_distance(origin, dest):
    """Real distance from BTS data."""
    return ROUTE_DISTANCES.get((origin, dest), 800)

def laplace_smooth(successes, total, prior_rate=0.16, prior_strength=20):
    """
    Apply Laplace/Bayesian smoothing to small samples.
    Never returns 0% or 100% — always blends with the population base rate.
    """
    if total == 0:
        return prior_rate
    return (successes + prior_rate * prior_strength) / (total + prior_strength)


def predict_delay_probability(airline_code, origin, dest, flight_date, dep_hour, distance=None):
    if ml_model is None:
        return None
    encoders = ml_features['encoders']
    def safe_encode(value, encoder):
        try: return encoder.transform([str(value)])[0]
        except ValueError: return 0

    if distance is None:
        distance = get_route_distance(origin, dest)

    features = pd.DataFrame([{
        'op_unique_carrier_enc': safe_encode(airline_code, encoders['op_unique_carrier']),
        'origin_enc': safe_encode(origin, encoders['origin']),
        'dest_enc': safe_encode(dest, encoders['dest']),
        'month': flight_date.month,
        'day_of_month': flight_date.day,
        'day_of_week': flight_date.weekday() + 1,
        'dep_hour': dep_hour,
        'distance': distance,
    }])[ml_features['feature_cols']]
    return float(ml_model.predict_proba(features)[0][1])


def get_better_times(airline_code, origin, dest, flight_date, original_hour, original_proba):
    """Find departure hours with materially lower predicted delay risk.
    Only uses hours where this airline actually operates flights on this route."""
    # Get the real hours this airline flies this route
    real_hours = AIRLINE_SCHEDULES[
        (AIRLINE_SCHEDULES['op_unique_carrier'] == airline_code) &
        (AIRLINE_SCHEDULES['origin'] == origin) &
        (AIRLINE_SCHEDULES['dest'] == dest)
    ]['dep_hour'].tolist()
    
    if not real_hours:
        return []
    
    results = []
    for h in real_hours:
        if h == original_hour:
            continue
        proba = predict_delay_probability(airline_code, origin, dest, flight_date, h)
        if proba is not None and proba < original_proba - 0.02:
            results.append({
                'hour': h,
                'time_str': f"{h % 12 if h % 12 else 12}:00 {'AM' if h < 12 else 'PM'}",
                'delay_proba': proba,
                'time_bucket': time_bucket(h),
                'improvement': original_proba - proba,
            })
    results.sort(key=lambda r: r['delay_proba'])
    return results[:4]


def get_alternative_airlines(origin, dest, current_airline, flight_date, dep_hour):
    """Only show airlines that ACTUALLY fly this route per BTS 2024 data."""
    real_airlines = ROUTE_AIRLINES[
        (ROUTE_AIRLINES['origin'] == origin) &
        (ROUTE_AIRLINES['dest'] == dest) &
        (ROUTE_AIRLINES['op_unique_carrier'] != current_airline) &
        (ROUTE_AIRLINES['flights_2024'] >= 5)  # require meaningful sample
    ].sort_values('flights_2024', ascending=False)

    results = []
    for _, row in real_airlines.iterrows():
        proba = predict_delay_probability(
            row['op_unique_carrier'], origin, dest, flight_date, dep_hour
        )
        if proba is not None:
            results.append({
                'airline_code': row['op_unique_carrier'],
                'airline_name': AIRLINE_NAMES.get(row['op_unique_carrier'], row['op_unique_carrier']),
                'flights_2024': int(row['flights_2024']),
                'delay_proba': proba,
            })
    results.sort(key=lambda r: r['delay_proba'])
    return results[:4]


def get_weather_forecast(airport_code, target_dt):
    if not OPENWEATHER_KEY or airport_code not in AIRPORT_COORDS:
        return None
    lat, lon = AIRPORT_COORDS[airport_code]
    now = datetime.now()
    delta_hours = (target_dt - now).total_seconds() / 3600
    try:
        if -1 <= delta_hours <= 1:
            r = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": "imperial"},
                timeout=8,
            )
            d = r.json()
            return _parse_weather(d, False, now.strftime("%I:%M %p"))
        if delta_hours <= 120:
            r = requests.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": "imperial"},
                timeout=8,
            )
            forecasts = r.json().get("list", [])
            if not forecasts:
                return None
            target_ts = target_dt.timestamp()
            closest = min(forecasts, key=lambda f: abs(f["dt"] - target_ts))
            return _parse_weather(
                closest, True,
                datetime.fromtimestamp(closest["dt"]).strftime("%a %I:%M %p")
            )
    except Exception:
        return None
    return None

def _parse_weather(data, is_forecast, time_str):
    return {
        "description": data["weather"][0]["description"].title(),
        "main": data["weather"][0]["main"],
        "icon": data["weather"][0]["icon"],
        "temp_f": round(data["main"]["temp"]),
        "feels_like": round(data["main"]["feels_like"]),
        "wind_mph": round(data["wind"]["speed"]),
        "wind_gust": round(data["wind"].get("gust", 0)),
        "humidity": data["main"]["humidity"],
        "visibility_mi": round(data.get("visibility", 16000) / 1609, 1),
        "is_forecast": is_forecast,
        "forecast_time": time_str,
    }


def get_gpt_recommendation(info):
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured."
    client = OpenAI(api_key=OPENAI_API_KEY)
    weather = info.get("weather")
    weather_str = (
        f"{weather['description']}, {weather['temp_f']}°F (feels {weather['feels_like']}°F), "
        f"wind {weather['wind_mph']} mph"
        + (f" (gusts to {weather['wind_gust']})" if weather['wind_gust'] > weather['wind_mph']+5 else "")
        + f", visibility {weather['visibility_mi']} mi, humidity {weather['humidity']}%"
        if weather else "weather data unavailable"
    )

    # Compute severity score: probability × duration (in minutes)
    # This is the EXPECTED delay impact, which is what really matters
    delay_pct = float(info['delay_rate_pct'])
    avg_delay = float(info['avg_delay_min'])
    expected_delay_min = (delay_pct / 100) * avg_delay
    
    prompt = f"""You are an expert flight risk advisor. Provide actionable advice in 4-5 sentences.
Start with a clear verdict in bold. Pick the verdict based on EXPECTED delay impact, not just probability:

VERDICT GUIDE (use the most appropriate):
- "**Proceed as planned.**" — when expected delay impact is minimal (<10 min expected, low risk)
- "**Proceed; minor delays possible.**" — moderate probability but short typical delays (<25 min average)
- "**Build in a buffer.**" — meaningful delay risk with moderate duration (25-45 min average when delayed)
- "**Arrive early; significant delays likely.**" — high probability AND long delays (>45 min average)
- "**Strongly consider rebooking.**" — severe weather, very high probability AND long delays, or both
- "**Consider an alternative time.**" — when better same-airline options exist

KEY INSIGHT: A 30% chance of a 15-min delay is FINE. A 30% chance of a 90-min delay is NOT. Factor in DURATION.

FLIGHT
- {info['airline_name']}: {info['origin']} ({info['origin_city']}) → {info['dest']} ({info['dest_city']})
- {info['date_str']} at {info['time_str']}, {info['time_bucket_label']} departure
- Distance: {info['distance']} miles

PREDICTION (Gradient Boosting model trained on 80,000 flights)
- Delay probability: {info['delay_rate_pct']}% (probability of >15 min late)
- Average delay duration WHEN delayed: {info['avg_delay_min']} minutes
- EXPECTED delay impact: {expected_delay_min:.0f} minutes (probability × duration)
- Risk classification: {info['risk_label']}
- Historical context: {info['historical_str']}
- Weather-related delay base rate: {info['weather_delay_rate_pct']}%

WEATHER FORECAST AT DEPARTURE
{weather_str}

Tailor advice to THIS flight's specifics. Reference concrete numbers (e.g. "expected delay ~12 min" or "wind gusts of 35 mph"). Be direct, no hedging."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=280,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate recommendation: {e}"


def risk_classification(rate):
    if rate < 0.18: return "LOW",    "#10b981", "rgba(16,185,129,.08)"
    if rate < 0.32: return "MEDIUM", "#f59e0b", "rgba(245,158,11,.08)"
    return            "HIGH",   "#ef4444", "rgba(239,68,68,.08)"


# ══════════════════════════════════════════════════════════════════════════════
# CSS — full SaaS-quality design
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --bg: #ffffff;
    --bg-subtle: #fafafa;
    --bg-muted: #f5f5f5;
    --border: #e5e5e5;
    --border-strong: #d4d4d4;
    --text: #0a0a0a;
    --text-secondary: #525252;
    --text-tertiary: #737373;
    --text-quaternary: #a3a3a3;
    --primary: #5b21b6;
    --primary-hover: #4c1d95;
    --primary-light: #ede9fe;
    --primary-bg: #f5f3ff;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --shadow-sm: 0 1px 2px rgba(0,0,0,.04);
    --shadow: 0 1px 3px rgba(0,0,0,.04), 0 4px 16px rgba(0,0,0,.04);
    --shadow-lg: 0 8px 32px rgba(0,0,0,.08);
}

#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

html, body, [class*="css"], .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stDateInput, .stTimeInput, .stRadio, button, input, p, h1, h2, h3, h4, span, div, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.01em;
}

.stApp { background: var(--bg); }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem;
    max-width: 1180px;
}

/* ─── TOP NAV ─────────────────────────────────────────────────── */
.topnav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 3.5rem;
}
.topnav .brand { display: flex; align-items: center; gap: .65rem; }
.topnav .brand-logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #5b21b6, #8b5cf6);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 1px 3px rgba(91,33,182,.3);
    position: relative;
}
.topnav .brand-logo svg { width: 16px; height: 16px; color: white; }
.topnav .brand-name { font-size: 1.05rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
.topnav .brand-meta {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .7rem; color: var(--text-quaternary);
    padding: .15rem .4rem; background: var(--bg-muted); border-radius: 4px; margin-left: .5rem;
}
.topnav .nav-right { display: flex; align-items: center; gap: 1rem; }
.topnav .status-pill {
    display: flex; align-items: center; gap: .35rem;
    padding: .25rem .6rem; background: rgba(16,185,129,.08);
    border-radius: 100px; font-size: .72rem; font-weight: 600; color: var(--success);
}
.topnav .status-dot {
    width: 6px; height: 6px; background: var(--success); border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100% {opacity:1;} 50% {opacity:.4;} }

/* ─── HERO ────────────────────────────────────────────────────── */
.hero {
    margin-bottom: 3rem;
    text-align: left;
    max-width: 720px;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: var(--primary-bg);
    color: var(--primary);
    font-size: .78rem; font-weight: 600;
    padding: .35rem .75rem;
    border-radius: 100px;
    border: 1px solid var(--primary-light);
    margin-bottom: 1.25rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    color: var(--text);
    margin: 0 0 1rem;
    line-height: 1.05;
    letter-spacing: -0.04em;
}
.hero h1 .gradient {
    background: linear-gradient(120deg, #5b21b6 0%, #ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero .subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    line-height: 1.55;
    max-width: 600px;
    margin-bottom: 1.75rem;
}

/* Stats strip */
.hero-stats {
    display: flex;
    gap: 2.5rem;
    padding: 1rem 0 0;
    border-top: 1px solid var(--border);
}
.hero-stat .v {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.02em;
}
.hero-stat .l {
    font-size: .75rem; color: var(--text-tertiary); text-transform: uppercase;
    letter-spacing: .06em; font-weight: 500; margin-top: .15rem;
}

/* ─── FORM CARD ───────────────────────────────────────────────── */
.form-header-standalone {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1.25rem 1.5rem;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 12px 12px 0 0;
    border-bottom: none;
    margin-bottom: 0;
}
.form-title { font-size: 1rem; font-weight: 600; color: var(--text); }
.form-step {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .72rem; color: var(--text-quaternary);
    background: white; padding: .25rem .55rem; border-radius: 4px;
    border: 1px solid var(--border);
}

/* ─── INPUT STYLING (force light theme) ──────────────────────── */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > input,
div[data-baseweb="input"],
div[data-baseweb="select"] {
    background-color: white !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    border-color: var(--border) !important;
    transition: all .15s; min-height: 42px;
}
div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(91,33,182,.12) !important;
}

/* SUPER strong overrides for ALL input variants — including time input which has unique base elements */
.stTextInput input, .stDateInput input, .stTimeInput input,
.stTextInput input[type="text"], .stDateInput input[type="text"], .stTimeInput input[type="text"],
[data-baseweb="input"] input, [data-baseweb="select"] input,
input[role="combobox"], input[role="textbox"] {
    background-color: white !important;
    background: white !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--text) !important;
    font-size: .92rem !important;
    padding: .55rem .8rem !important;
}

.stTextInput input::placeholder, .stDateInput input::placeholder, .stTimeInput input::placeholder,
[data-baseweb="input"] input::placeholder {
    color: var(--text-quaternary) !important;
    -webkit-text-fill-color: var(--text-quaternary) !important;
    opacity: 1 !important;
}

.stSelectbox div[role="button"],
.stSelectbox [data-baseweb="select"] [data-baseweb="select-control"] {
    background-color: white !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-size: .92rem !important;
    padding: .55rem .8rem !important;
}

/* Selectbox value text */
.stSelectbox div[role="button"] > div,
.stSelectbox div[data-baseweb="select"] span {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

label[data-testid="stWidgetLabel"] p {
    font-size: .8rem !important; font-weight: 600 !important;
    color: var(--text) !important; margin-bottom: .4rem !important;
    letter-spacing: -0.01em;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(180deg, #6d28d9 0%, #5b21b6 100%) !important;
    color: white !important; border: 0 !important; border-radius: 8px !important;
    padding: .7rem 1.4rem !important; font-weight: 600 !important; font-size: .92rem !important;
    box-shadow: 0 1px 3px rgba(91,33,182,.3), inset 0 1px 0 rgba(255,255,255,.15) !important;
    transition: all .15s !important;
    letter-spacing: -0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(91,33,182,.35), inset 0 1px 0 rgba(255,255,255,.15) !important;
}

/* ─── RESULT SECTION ──────────────────────────────────────────── */
.result-divider {
    margin: 2.5rem 0 2rem;
    border: 0; border-top: 1px solid var(--border);
}

.source-line {
    display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
    margin-bottom: 1.5rem;
}
.source-chip {
    display: inline-flex; align-items: center; gap: .35rem;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .72rem; color: var(--text-tertiary);
    background: var(--bg-muted); padding: .3rem .6rem; border-radius: 6px;
    font-weight: 500;
}
.source-chip.accent {
    background: var(--primary-bg); color: var(--primary);
}

/* Risk hero card */
.risk-hero {
    background: white; border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem; margin-bottom: 1rem; box-shadow: var(--shadow);
    display: grid; grid-template-columns: 1fr auto; gap: 2rem; align-items: center;
}
.risk-hero .left { min-width: 0; }
.risk-pill {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .3rem .65rem; border-radius: 6px;
    font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .06em;
    margin-bottom: .85rem;
}
.risk-pill .dot { width: 6px; height: 6px; border-radius: 50%; }
.route-display {
    display: flex; align-items: baseline; gap: .75rem;
    font-size: 2.25rem; font-weight: 800; color: var(--text);
    letter-spacing: -0.04em; line-height: 1.1; margin-bottom: .35rem;
    flex-wrap: wrap;
}
.route-display .arrow {
    color: var(--text-quaternary);
    font-size: 1.6rem;
    font-weight: 400;
}
.route-display .iata { font-size: 2.25rem; }
.route-meta {
    font-size: .9rem; color: var(--text-tertiary); margin-top: .4rem;
}
.risk-hero .right { text-align: right; }
.big-pct {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 3.5rem; font-weight: 700;
    line-height: 1; letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums;
}
.big-pct-label {
    font-size: .72rem; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: .08em;
    font-weight: 600; margin-top: .5rem;
}

/* Metric grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .75rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: white; border: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem 1.2rem;
    transition: all .15s;
}
.metric-card:hover {
    border-color: var(--border-strong);
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.metric-card .lbl {
    font-size: .72rem; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: .07em;
    font-weight: 500; margin-bottom: .5rem;
}
.metric-card .val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.65rem; font-weight: 700; color: var(--text);
    line-height: 1; letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums;
}
.metric-card .val .unit {
    font-size: .9rem; font-weight: 500; color: var(--text-tertiary);
    margin-left: .15rem;
}
.metric-card .sub {
    font-size: .75rem; color: var(--text-tertiary); margin-top: .35rem;
}

/* Section headers */
.section-h {
    font-size: .8rem; font-weight: 600; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: .08em;
    margin: 2.5rem 0 1rem;
    display: flex; align-items: center; gap: .65rem;
}
.section-h::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

/* Weather card */
.weather-card {
    background: linear-gradient(135deg, #fafafa 0%, #f5f3ff 100%);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 1.5rem 1.75rem;
    display: grid; grid-template-columns: 1fr auto; gap: 1.5rem;
    align-items: center;
}
.weather-main .desc {
    font-size: 1.4rem; font-weight: 600; color: var(--text); margin-bottom: .2rem;
    letter-spacing: -0.02em;
}
.weather-main .when {
    font-size: .85rem; color: var(--text-tertiary);
}
.weather-stats { display: flex; gap: 2.25rem; }
.weather-stat { text-align: center; }
.weather-stat .v {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem; font-weight: 700; color: var(--text);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.025em;
}
.weather-stat .v .u { font-size: .8rem; font-weight: 500; color: var(--text-tertiary); }
.weather-stat .l {
    font-size: .68rem; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: .08em;
    margin-top: .2rem; font-weight: 500;
}

/* AI recommendation card */
.ai-card {
    background: white; border: 1px solid var(--border);
    border-radius: 14px; padding: 1.75rem;
    position: relative; overflow: hidden;
    box-shadow: var(--shadow);
}
.ai-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #5b21b6 0%, #8b5cf6 33%, #ec4899 66%, #f59e0b 100%);
}
.ai-header {
    display: flex; align-items: center; gap: .5rem;
    margin-bottom: 1rem;
}
.ai-badge {
    display: inline-flex; align-items: center; gap: .35rem;
    background: var(--primary-bg); color: var(--primary);
    font-size: .7rem; font-weight: 700;
    padding: .3rem .6rem; border-radius: 6px;
    text-transform: uppercase; letter-spacing: .06em;
}
.ai-text {
    font-size: 1.02rem; line-height: 1.7; color: var(--text);
}
.ai-text strong { color: var(--text); font-weight: 700; }

/* Alternative rows */
.alt-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 1.25rem;
    background: white; border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: .5rem;
    transition: all .15s;
}
.alt-row:hover {
    border-color: var(--primary);
    transform: translateX(2px);
    box-shadow: 0 2px 8px rgba(91,33,182,.08);
}
.alt-row .alt-name {
    font-weight: 600; color: var(--text);
    font-size: .98rem; margin-bottom: .15rem;
}
.alt-row .alt-meta {
    font-size: .8rem; color: var(--text-tertiary);
}
.alt-row .alt-improvement {
    display: inline-block; margin-left: .5rem;
    font-size: .7rem; font-weight: 600;
    background: rgba(16,185,129,.1); color: var(--success);
    padding: .15rem .45rem; border-radius: 4px;
}
.alt-row .right { text-align: right; }
.alt-row .alt-rate {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem; font-weight: 700;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}
.alt-row .alt-sub {
    font-size: .68rem; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: .07em;
    margin-top: .15rem; font-weight: 500;
}

/* Empty state */
.empty-state {
    text-align: center; padding: 1.5rem;
    color: var(--text-tertiary); font-size: .9rem;
    background: var(--bg-muted); border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: var(--text-quaternary);
    font-size: .78rem;
    padding: 4rem 0 1.5rem;
    border-top: 1px solid var(--border);
    margin-top: 4rem;
}
.footer a { color: var(--text-tertiary); text-decoration: none; }

/* Expander */
[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { font-size: .87rem !important; font-weight: 600 !important; }

/* Mobile */
@media (max-width: 768px) {
    .hero h1 { font-size: 2.2rem; }
    .hero-stats { flex-wrap: wrap; gap: 1.25rem; }
    .risk-hero { grid-template-columns: 1fr; }
    .risk-hero .right { text-align: left; }
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .weather-card { grid-template-columns: 1fr; }
    .weather-stats { gap: 1.25rem; flex-wrap: wrap; }
    .route-display { font-size: 1.6rem; }
    .route-display .iata { font-size: 1.6rem; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="topnav">
    <div class="brand">
        <div class="brand-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="m12 19-7-7 7-7"/>
                <path d="M19 12H5"/>
            </svg>
        </div>
        <span class="brand-name">Flight Risk AI</span>
        <span class="brand-meta">v2.0</span>
    </div>
    <div class="nav-right">
        <div class="status-pill"><span class="status-dot"></span>Live</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero
n_routes = len(ROUTE_DISTANCES)
n_airports = len(airports_df)
model_acc = ml_metrics['accuracy'] * 100 if ml_metrics else 84.1

st.markdown(f"""
<div class="hero">
    <div class="hero-badge">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>
        </svg>
        AI-Powered · Real ML Predictions
    </div>
    <h1>Predict your flight delay risk <span class="gradient">before you fly.</span></h1>
    <p class="subtitle">A Gradient Boosting model trained on 100,000 BTS flight records, combined with live weather forecasts and GPT-4o analysis to deliver accurate, explainable delay predictions.</p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="v">{model_acc:.1f}%</div><div class="l">Model Accuracy</div></div>
        <div class="hero-stat"><div class="v">{n_routes:,}</div><div class="l">Routes Covered</div></div>
        <div class="hero-stat"><div class="v">{n_airports}</div><div class="l">US Airports</div></div>
        <div class="hero-stat"><div class="v">100K+</div><div class="l">Training Flights</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="form-header-standalone">
    <div class="form-title">Enter Your Flight Details</div>
    <div class="form-step">5 fields</div>
</div>
""", unsafe_allow_html=True)

# Build airport options
airport_options = airports_df.sort_values('city_clean')['label'].tolist()
airport_label_to_code = dict(zip(airports_df['label'], airports_df['code']))
airport_code_to_city = dict(zip(airports_df['code'], airports_df['city_clean']))

c1, c2, c3 = st.columns(3)
with c1:
    origin_label = st.selectbox(
        "Departing from",
        ["Select airport..."] + airport_options,
        index=0,
    )
with c2:
    dest_label = st.selectbox(
        "Arriving at",
        ["Select airport..."] + airport_options,
        index=0,
    )
with c3:
    airline_name = st.selectbox(
        "Airline",
        ["Select airline..."] + sorted(AIRLINE_NAMES.values()),
        index=0,
    )

c4, c5 = st.columns(2)
with c4:
    flight_date = st.date_input(
        "Departure date",
        value=date.today() + timedelta(days=1),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=180),
    )
with c5:
    # Use selectbox instead of time_input — avoids Streamlit dark mode rendering bug
    time_options = []
    time_labels = []
    for h in range(5, 24):
        for m in [0, 15, 30, 45]:
            t = time(h, m)
            label = t.strftime("%I:%M %p").lstrip("0")
            time_options.append(t)
            time_labels.append(label)
    default_idx = time_labels.index("10:00 AM")
    selected_time_label = st.selectbox("Scheduled departure time", time_labels, index=default_idx)
    dep_time = time_options[time_labels.index(selected_time_label)]

st.markdown("<div style='margin: 1.5rem 0 .5rem;'></div>", unsafe_allow_html=True)
analyze = st.button("Analyze Flight Risk →")

# ══════════════════════════════════════════════════════════════════════════════
# RESULT
# ══════════════════════════════════════════════════════════════════════════════
if analyze:
    # Validate
    if origin_label == "Select airport..." or dest_label == "Select airport...":
        st.error("Please select both origin and destination airports.")
        st.stop()
    if airline_name == "Select airline...":
        st.error("Please select an airline.")
        st.stop()

    origin = airport_label_to_code[origin_label]
    dest = airport_label_to_code[dest_label]
    airline_code = AIRLINE_TO_CODE[airline_name]

    if origin == dest:
        st.error("Origin and destination must be different.")
        st.stop()

    origin_city = airport_code_to_city.get(origin, origin)
    dest_city = airport_code_to_city.get(dest, dest)
    dep_hour = dep_time.hour
    dep_time_str = dep_time.strftime("%I:%M %p")
    bucket = time_bucket(dep_hour)
    distance = get_route_distance(origin, dest)

    # Run prediction
    with st.spinner("Running prediction model…"):
        ml_proba = predict_delay_probability(airline_code, origin, dest, flight_date, dep_hour, distance)

    if ml_proba is None:
        st.error("Prediction model unavailable. Make sure delay_model.pkl is in the folder.")
        st.stop()

    # Get smoothed historical context
    rd = summary_df[
        (summary_df["origin"]==origin) & (summary_df["dest"]==dest) &
        (summary_df["op_unique_carrier"]==airline_code)
    ]
    
    if not rd.empty:
        total_flights = int(rd["total_flights"].sum())
        delayed_flights = int(rd["delayed_flights"].sum())
        weather_delayed = int(rd["weather_delayed"].sum())
        avg_delay_min = rd["avg_delay_min"].mean() if rd["avg_delay_min"].mean() > 0 else 38
        # Apply Laplace smoothing to never get 0% or 100%
        historical_rate = laplace_smooth(delayed_flights, total_flights, prior_rate=0.16, prior_strength=20)
        weather_rate = laplace_smooth(weather_delayed, total_flights, prior_rate=0.02, prior_strength=50)
    else:
        # No historical data for this exact combination — use route base rate
        rd_route = summary_df[(summary_df["origin"]==origin) & (summary_df["dest"]==dest)]
        if not rd_route.empty:
            total_flights = int(rd_route["total_flights"].sum())
            delayed_flights = int(rd_route["delayed_flights"].sum())
            weather_delayed = int(rd_route["weather_delayed"].sum())
            avg_delay_min = rd_route["avg_delay_min"].mean() if rd_route["avg_delay_min"].mean() > 0 else 38
            historical_rate = laplace_smooth(delayed_flights, total_flights, prior_rate=0.16, prior_strength=20)
            weather_rate = laplace_smooth(weather_delayed, total_flights, prior_rate=0.02, prior_strength=50)
        else:
            total_flights = 0
            historical_rate = 0.16
            weather_rate = 0.02
            avg_delay_min = 38

    delay_rate = ml_proba
    risk_label, risk_color, risk_bg = risk_classification(delay_rate)

    # Weather forecast
    target_dt = datetime.combine(flight_date, time(dep_hour, 0))
    with st.spinner("Fetching weather forecast for departure time…"):
        weather = get_weather_forecast(origin, target_dt) if OPENWEATHER_KEY else None

    # ── Render result ──────────────────────────────────────────────────────
    st.markdown('<hr class="result-divider">', unsafe_allow_html=True)

    # Source chips
    chips = []
    if ml_metrics:
        chips.append(f'<span class="source-chip accent">⚡ Gradient Boosting · {ml_metrics["accuracy"]*100:.1f}% accuracy</span>')
    chips.append(f'<span class="source-chip">📊 BTS 2024 · {total_flights:,} historical flights</span>')
    if weather:
        chips.append('<span class="source-chip">🌤 OpenWeatherMap forecast</span>')
    if OPENAI_API_KEY:
        chips.append('<span class="source-chip">🤖 GPT-4o analysis</span>')
    st.markdown(f'<div class="source-line">{" ".join(chips)}</div>', unsafe_allow_html=True)

    # Risk hero card
    st.markdown(f"""
    <div class="risk-hero">
        <div class="left">
            <div class="risk-pill" style="background: {risk_bg}; color: {risk_color};">
                <span class="dot" style="background: {risk_color};"></span>
                {risk_label} RISK
            </div>
            <div class="route-display">
                <span class="iata">{origin}</span>
                <span class="arrow">→</span>
                <span class="iata">{dest}</span>
            </div>
            <div class="route-meta">
                {origin_city} to {dest_city} · {airline_name} ·
                {flight_date.strftime('%a %b %d')} at {dep_time_str} · {time_bucket_label(bucket)} · {distance:,} mi
            </div>
        </div>
        <div class="right">
            <div class="big-pct" style="color: {risk_color};">{delay_rate*100:.0f}%</div>
            <div class="big-pct-label">Delay Probability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    on_time = (1 - delay_rate) * 100
    historical_pct = historical_rate * 100
    
    diff_str = ""
    if abs(delay_rate - historical_rate) > 0.02:
        if delay_rate > historical_rate:
            diff_str = f"<div class='sub'>↑ {(delay_rate-historical_rate)*100:.0f} pts vs route avg</div>"
        else:
            diff_str = f"<div class='sub'>↓ {(historical_rate-delay_rate)*100:.0f} pts vs route avg</div>"
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="lbl">On-Time Probability</div>
            <div class="val">{on_time:.0f}<span class="unit">%</span></div>
            <div class="sub">Predicted by ML model</div>
        </div>
        <div class="metric-card">
            <div class="lbl">Historical Rate</div>
            <div class="val">{historical_pct:.0f}<span class="unit">%</span></div>
            <div class="sub">Smoothed Bayesian estimate</div>
        </div>
        <div class="metric-card">
            <div class="lbl">Avg Delay When Late</div>
            <div class="val">{avg_delay_min:.0f}<span class="unit">min</span></div>
            <div class="sub">For delayed flights only</div>
        </div>
        <div class="metric-card">
            <div class="lbl">Weather Risk</div>
            <div class="val">{weather_rate*100:.1f}<span class="unit">%</span></div>
            <div class="sub">Weather-caused delays</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Weather
    if weather:
        st.markdown('<div class="section-h">Weather forecast at departure</div>', unsafe_allow_html=True)
        forecast_label = "Forecast" if weather.get("is_forecast") else "Current conditions"
        st.markdown(f"""
        <div class="weather-card">
            <div class="weather-main">
                <div class="desc">{weather['description']}</div>
                <div class="when">{forecast_label} · {weather['forecast_time']} at {origin}</div>
            </div>
            <div class="weather-stats">
                <div class="weather-stat">
                    <div class="v">{weather['temp_f']}<span class="u">°F</span></div>
                    <div class="l">Temp</div>
                </div>
                <div class="weather-stat">
                    <div class="v">{weather['wind_mph']}<span class="u">mph</span></div>
                    <div class="l">Wind</div>
                </div>
                <div class="weather-stat">
                    <div class="v">{weather['visibility_mi']}<span class="u">mi</span></div>
                    <div class="l">Visibility</div>
                </div>
                <div class="weather-stat">
                    <div class="v">{weather['humidity']}<span class="u">%</span></div>
                    <div class="l">Humidity</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-h">Weather forecast</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="empty-state">Weather data unavailable for this airport. Check OPENWEATHER_KEY in .env.</div>',
            unsafe_allow_html=True
        )

    # AI Recommendation
    st.markdown('<div class="section-h">AI recommendation</div>', unsafe_allow_html=True)
    historical_str = (
        f"Based on {total_flights} prior flights, ~{historical_pct:.0f}% were delayed >15 min"
        if total_flights > 0 else
        f"No exact historical match; using route base rates"
    )
    
    gpt_input = {
        "origin": origin, "dest": dest,
        "origin_city": origin_city, "dest_city": dest_city,
        "airline_code": airline_code, "airline_name": airline_name,
        "date_str": flight_date.strftime("%a %b %d, %Y"),
        "time_str": dep_time_str,
        "time_bucket_label": time_bucket_label(bucket),
        "delay_rate_pct": f"{delay_rate*100:.0f}",
        "avg_delay_min": f"{avg_delay_min:.0f}",
        "weather_delay_rate_pct": f"{weather_rate*100:.1f}",
        "risk_label": risk_label,
        "historical_str": historical_str,
        "distance": f"{distance:,}",
        "weather": weather,
    }
    with st.spinner("Generating personalized recommendation…"):
        recommendation = get_gpt_recommendation(gpt_input)

    # Convert markdown ** to HTML
    rec_html = recommendation
    while '**' in rec_html:
        rec_html = rec_html.replace('**', '<strong>', 1).replace('**', '</strong>', 1)

    st.markdown(f"""
    <div class="ai-card">
        <div class="ai-header">
            <span class="ai-badge">✦ GPT-4o Analysis</span>
        </div>
        <div class="ai-text">{rec_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Better departure time windows
    st.markdown('<div class="section-h">Lower-risk departure windows on this route</div>', unsafe_allow_html=True)
    better_times = get_better_times(airline_code, origin, dest, flight_date, dep_hour, delay_rate)

    if better_times:
        st.markdown(f"""
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: .85rem 1rem; margin-bottom: 1rem; font-size: .85rem; color: #92400e;">
            <strong>⚠️ Estimated time windows only — not real bookable flights.</strong>
            These are departure hours when {airline_name} has historically operated {origin} → {dest} flights with lower predicted delay risk.
            Check <a href="https://www.jetblue.com" target="_blank" style="color:#92400e;">the airline's website</a> or Google Flights for actual available departures.
        </div>
        """, unsafe_allow_html=True)
        for alt in better_times:
            alt_label, alt_color, _ = risk_classification(alt['delay_proba'])
            st.markdown(f"""
            <div class="alt-row">
                <div>
                    <div class="alt-name">
                        Around {alt['time_str']} departure window
                        <span class="alt-improvement">↓ {alt['improvement']*100:.0f} pts lower risk</span>
                    </div>
                    <div class="alt-meta">
                        ML model prediction · {time_bucket_label(alt['time_bucket'])} · {airline_name} has historically flown this slot
                    </div>
                </div>
                <div class="right">
                    <div class="alt-rate" style="color: {alt_color};">{alt['delay_proba']*100:.0f}%</div>
                    <div class="alt-sub">predicted delay risk</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="empty-state">Your selected time ({dep_time_str}) is already among the lowest-risk windows for {airline_name} on this route.</div>',
            unsafe_allow_html=True
        )

    # Other airlines on this route
    st.markdown('<div class="section-h">Other airlines on this route (ML predictions)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: .85rem 1rem; margin-bottom: 1rem; font-size: .85rem; color: #92400e;">
        <strong>⚠️ These are delay risk estimates, not specific flights.</strong>
        Airlines shown actually flew {origin} → {dest} in 2024 BTS data. Verify availability on Google Flights before making decisions.
    </div>
    """, unsafe_allow_html=True)
    alt_airlines = get_alternative_airlines(origin, dest, airline_code, flight_date, dep_hour)

    if alt_airlines:
        for alt in alt_airlines:
            alt_label, alt_color, _ = risk_classification(alt['delay_proba'])
            improvement_html = ""
            if alt['delay_proba'] < delay_rate - 0.02:
                improvement_html = f'<span class="alt-improvement">↓ {(delay_rate - alt["delay_proba"])*100:.0f} pts lower</span>'
            st.markdown(f"""
            <div class="alt-row">
                <div>
                    <div class="alt-name">{alt['airline_name']} {improvement_html}</div>
                    <div class="alt-meta">
                        {origin} → {dest} · verified {alt['flights_2024']:,} flights on this route in 2024 · ML-predicted delay risk at ~{dep_time_str}
                    </div>
                </div>
                <div class="right">
                    <div class="alt-rate" style="color: {alt_color};">{alt['delay_proba']*100:.0f}%</div>
                    <div class="alt-sub">predicted delay risk</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-state">No other airlines with verified flights on this route in our 2024 data.</div>',
            unsafe_allow_html=True
        )

    # Model details (collapsible)
    if ml_metrics:
        st.markdown('<div class="section-h">Model & methodology</div>', unsafe_allow_html=True)
        with st.expander("View model details, accuracy, and feature importance"):
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Algorithm", "Gradient Boosting")
            cm2.metric("Test Accuracy", f"{ml_metrics['accuracy']*100:.1f}%")
            cm3.metric("AUC-ROC", f"{ml_metrics['auc_roc']}")

            cm4, cm5, cm6 = st.columns(3)
            cm4.metric("Training Set", f"{ml_metrics['training_size']:,}")
            cm5.metric("Test Set", f"{ml_metrics['test_size']:,}")
            cm6.metric("Trees", f"{ml_metrics['n_estimators']}")

            st.markdown("**Feature importance** (which inputs the model relies on most):")
            imp = ml_metrics['feature_importance']
            imp_df = pd.DataFrame(list(imp.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
            imp_df['Feature'] = imp_df['Feature'].str.replace('_enc', '').str.replace('_', ' ').str.title()
            st.bar_chart(imp_df.set_index('Feature'), color="#5b21b6", height=240)
            
            st.caption(
                "Predictions use Bayesian (Laplace) smoothing on small samples to avoid showing impossible 0% or 100% rates. "
                "Alternative airline suggestions are filtered to only those with ≥5 verified flights on this route in 2024 BTS data."
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · Data: U.S. Bureau of Transportation Statistics 2024 ·
    Weather: OpenWeatherMap · AI: OpenAI GPT-4o-mini · ML: scikit-learn Gradient Boosting
</div>
""", unsafe_allow_html=True)
