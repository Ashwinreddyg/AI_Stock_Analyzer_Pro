import os
import math
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import logging
from pathlib import Path
import streamlit as st
import yfinance as yf

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

# ============================================================
# AI Stock Analyzer Pro v8.6
# ============================================================
st.set_page_config(
    page_title="AI Stock Analyzer Pro v8.6",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "8.6"

def cfg(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()

BENZINGA_KEY = cfg("BENZINGA_API_KEY")
OPENAI_KEY = cfg("OPENAI_API_KEY")
ANTHROPIC_KEY = cfg("ANTHROPIC_API_KEY")
OLLAMA_URL = cfg("OLLAMA_URL", "http://localhost:11434/api/generate")
OPENAI_MODEL_DEFAULT = cfg("OPENAI_MODEL", "gpt-5")
CLAUDE_MODEL_DEFAULT = cfg("CLAUDE_MODEL", "claude-opus-5")

LOG_DIR = Path("logs")
REPORT_DIR = Path("data") / "daily_reports"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = Path("data") / "recommendation_history.csv"
DAILY_STATE_FILE = Path("data") / "daily_intelligence_state.json"
logging.basicConfig(filename=LOG_DIR / "app.log", level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ai_stock_analyzer")
BENZINGA_BASE = "https://api.benzinga.com/api/v2.1"

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "ORCL", "AMD",
    "NFLX", "CRM", "PLTR", "MU", "TSM", "QCOM", "INTC", "SMCI", "SOFI", "RIVN",
    "TTD", "GTLB", "DUOL", "OKLO", "SOUN", "BBAI", "AI", "PATH", "FIVN", "CALX",
]
MARKET_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "VIXY", "GLD", "USO", "TLT"]

# Free scanner universe: liquid US equities. Price/volume scanning uses yfinance directly.
# The scanner can be expanded with a custom comma-separated universe in the UI.
SCANNER_UNIVERSE = [
"AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","ORCL","AMD","NFLX","CRM","PLTR","MU","TSM","QCOM","INTC","SMCI","SOFI","RIVN","TTD","GTLB","DUOL","OKLO","SOUN","BBAI","AI","PATH","FIVN","CALX",
"ADBE","CSCO","IBM","NOW","INTU","TXN","AMAT","LRCX","KLAC","ADI","MELI","PANW","CRWD","SNOW","DDOG","NET","MDB","SHOP","UBER","ABNB","COIN","HOOD","PYPL","SQ","AFRM","NU","SOFI","IONQ","RKLB","ASTS","HIMS","TEM","CRSP","RXRX","BEAM","DNA","IONQ","ACHR","JOBY",
"JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","V","MA","COF","USB","PNC","TFC","BK","CME","ICE","SPGI","MCO","CB","AON","MMC","PGR","ALL","TRV","MET","PRU",
"UNH","LLY","JNJ","MRK","ABBV","PFE","BMY","AMGN","GILD","REGN","VRTX","ISRG","MDT","SYK","BSX","EW","DHR","TMO","ABT","ELV","CI","CVS","HCA","ZBH","BDX","IDXX","DXCM","RMD",
"XOM","CVX","COP","EOG","SLB","OXY","MPC","PSX","VLO","HAL","DVN","FANG","KMI","WMB","OKE","ET","LNG","BKR","CTRA","APA",
"CAT","DE","GE","HON","RTX","BA","LMT","NOC","GD","UPS","FDX","UNP","CSX","WM","ETN","EMR","PH","ITW","MMM","CARR","JCI","TT","URI","FAST","PCAR","CMI","ROK","AME","VRSK",
"WMT","COST","TGT","HD","LOW","TJX","NKE","SBUX","MCD","CMG","YUM","KO","PEP","PM","MO","PG","CL","EL","KHC","MDLZ","GIS","K","HSY","MNST","KR","DG","DLTR","ORLY","AZO","ULTA","LULU","DECK","DHI","LEN","PHM",
"DIS","CMCSA","T","VZ","TMUS","CHTR","NFLX","WBD","SPOT","ROKU","FOXA","LYV","EA","TTWO","RBLX","U","MTCH","PINS","SNAP","RDDT",
"LIN","APD","SHW","ECL","FCX","NEM","NUE","STLD","DOW","DD","ALB","VMC","MLM","CE","PPG","IFF","BALL","IP","AVY","MOS","CF","FMC",
"PLD","AMT","EQIX","CCI","O","SPG","DLR","PSA","WELL","VICI","AVB","EQR","ESS","ARE","CBRE","CSGP","IRM",
"ACN","CTSH","EPAM","GEN","HPE","HPQ","DELL","STX","WDC","ARM","MRVL","ON","NXPI","MCHP","MPWR","TER","SNPS","CDNS","ANSS","FTNT","ZS","OKTA","CHKP","AKAM","TTD","APP","GDDY","FSLY","ESTC","CFLT","TWLO","DOCU","ZM","VEEV","PAYC","HUBS","TEAM","WDAY","ASML","SAP","SONY","BABA","PDD","JD","BIDU","NTES","NIO","LI","XPEV","SE","MSTR","MARA","RIOT","CLSK","IBKR","RKT","HOPE",
"SPY","QQQ","DIA","IWM","GLD","TLT","XLF","XLK","XLE","XLV","XLY","XLI","XLC","XLU","XLP","XLB","XLRE"
]
SCANNER_UNIVERSE = list(dict.fromkeys(SCANNER_UNIVERSE))

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "NVDA"
if "compare_tickers" not in st.session_state:
    st.session_state.compare_tickers = "NVDA,AAPL,MSFT,AVGO,AMD"
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "hide_right_rail" not in st.session_state:
    st.session_state.hide_right_rail = False
if "focus_mode" not in st.session_state:
    st.session_state.focus_mode = False
if "daily_scan_done" not in st.session_state:
    st.session_state.daily_scan_done = False
if "daily_ai_report" not in st.session_state:
    st.session_state.daily_ai_report = ""
if "daily_scan_df" not in st.session_state:
    st.session_state.daily_scan_df = pd.DataFrame()
if "stock_detail_open" not in st.session_state:
    st.session_state.stock_detail_open = False

def log_stock_event(ticker: str, action: str = "Selected"):
    ticker = str(ticker).upper().strip()
    if not ticker:
        return
    entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "ticker": ticker, "action": action}
    if not st.session_state.activity_log or st.session_state.activity_log[-1] != entry:
        st.session_state.activity_log.append(entry)
        st.session_state.activity_log = st.session_state.activity_log[-100:]
    try:
        logger.info("%s | %s", action, ticker)
    except Exception:
        pass

# ============================================================
# Premium light-blue styling
# ============================================================
st.markdown(
    """
<style>
:root {
  --navy:#123a6b; --blue:#1687f8; --blue2:#3aa7ff; --pale:#eef7ff;
  --line:#d9e9f8; --text:#173b64; --muted:#6d829a; --green:#12a878;
  --red:#e05260; --gold:#f0a21a; --white:#ffffff;
}
.stApp { background:linear-gradient(180deg,#f6fbff 0%,#eef7ff 100%); color:var(--text); }
.block-container { padding:0.65rem 1rem 2rem; max-width:1800px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#eef8ff,#e6f3ff); border-right:1px solid #cfe4f7; }
[data-testid="stSidebar"] .block-container { padding:1rem 0.85rem; }
.header {background:rgba(255,255,255,.96);border:1px solid #d7e9f8;border-radius:20px;padding:14px 16px;box-shadow:0 10px 32px rgba(23,76,125,.09);margin-bottom:10px;}
.brand-wrap{display:flex;align-items:center;gap:11px;min-height:54px;}
.logo-mark{width:38px;height:38px;border-radius:12px;position:relative;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,#0b6fd3,#38a9ff);box-shadow:0 7px 18px rgba(22,135,248,.25);overflow:hidden;color:#fff;font-size:10px;font-weight:950;letter-spacing:-.5px;flex:0 0 auto;}
.logo-mark:before{content:'';position:absolute;inset:5px;border:1px solid rgba(255,255,255,.3);border-radius:9px;}
.logo-mark span{position:absolute;bottom:7px;width:4px;border-radius:4px;background:#fff;opacity:.92;}
.logo-mark span:nth-child(1){height:9px;left:9px}.logo-mark span:nth-child(2){height:15px;left:16px}.logo-mark span:nth-child(3){height:21px;left:23px}
.logo-mark b{position:relative;z-index:2;font-size:9px;margin-left:4px;margin-top:-18px;}
.side-brand{display:flex;align-items:center;gap:10px;padding:5px 3px 14px;margin-bottom:10px;border-bottom:1px solid #d6e8f6}.side-brand-title{font-size:17px;font-weight:900;color:#123a6b;letter-spacing:-.35px}.side-brand-sub{font-size:9px;color:#7890a8;margin-top:2px}.logo-large{width:46px;height:46px;border-radius:14px;}
.logo-large b{font-size:10px;}
.brand {font-size:23px;font-weight:900;color:#123a6b;letter-spacing:-.7px;line-height:1.05;}
 .brand-wrap{background:linear-gradient(135deg,#ffffff,#f2f9ff);border:1px solid #d8eaf8;border-radius:18px;padding:8px 12px;box-shadow:0 6px 20px rgba(22,95,150,.06);}
 .logo-mark{background:linear-gradient(145deg,#075fc0,#35a9ff);}
 .forecast-shell{background:linear-gradient(135deg,#ffffff,#f4faff);border:1px solid #d5e8f7;border-radius:18px;padding:14px;box-shadow:0 5px 18px rgba(28,88,140,.05);}
 .forecast-title{font-size:13px;font-weight:900;color:#163f6b;margin-bottom:3px;}
 .forecast-sub{font-size:10px;color:#7890a8;margin-bottom:10px;}
 .edu-card{background:linear-gradient(135deg,#ffffff,#f5fbff);border:1px solid #d8eaf8;border-radius:16px;padding:15px;height:100%;box-shadow:0 5px 16px rgba(28,88,140,.04);}
 .edu-term{font-size:12px;font-weight:850;color:#174a77;margin-bottom:4px;}
 .edu-def{font-size:11px;color:#536f88;line-height:1.55;}
.brand small {display:block;font-size:10px;color:#7189a3;font-weight:600;letter-spacing:.15px;margin-top:5px;}
.status-pill{height:52px;border:1px solid #d7e8f6;border-radius:13px;background:#f7fbff;padding:8px 10px;display:flex;flex-direction:column;align-items:flex-start;justify-content:center;color:#234e76;font-size:12px;}
.status-pill small{font-size:9px;color:#7a91a8;margin-top:2px}.live-dot{width:7px;height:7px;border-radius:50%;background:#11aa78;display:inline-block;box-shadow:0 0 0 4px #e6f9f2;margin-bottom:3px;}
.searchbox {border:1px solid #cfe1f3;border-radius:10px;padding:10px 13px;background:#f8fcff;color:#7b91a8;font-size:12px;}
.market-strip {display:flex;gap:0;background:#fff;border:1px solid var(--line);border-radius:13px;overflow:hidden;margin-bottom:10px;}
.market-cell {flex:1;padding:10px 12px 7px;border-right:1px solid #e4eef7;min-width:100px;background:linear-gradient(180deg,#fff,#fbfdff);transition:all .18s ease;}
.market-cell:hover{background:#f2f9ff;transform:translateY(-1px);}
.market-cell:last-child{border-right:0}.market-name{font-size:10px;color:#6f849b;text-transform:uppercase;font-weight:750;letter-spacing:.35px}.market-value{font-size:15px;font-weight:900;color:#183e67;margin-top:2px}.market-change{font-size:10px;font-weight:800;margin-top:1px}
.card {background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px 16px;box-shadow:0 5px 18px rgba(28,88,140,.055);margin-bottom:12px;}
.card-title{font-size:14px;font-weight:800;color:#163e69;margin-bottom:7px}.card-sub{font-size:11px;color:#7890a8}
.hero-card{background:linear-gradient(120deg,#fff 0%,#f5fbff 100%);border:1px solid #cfe6fa;border-radius:18px;padding:17px 18px;box-shadow:0 7px 24px rgba(36,113,177,.08);}
.hero-ticker{font-size:25px;font-weight:900;color:#123a6b}.hero-company{font-size:12px;color:#7389a0}.hero-price{font-size:29px;font-weight:900;color:#164a7c;margin-top:4px}.hero-up{color:#0aa878;font-weight:800}.hero-down{color:#e05260;font-weight:800}
.kpi {background:#fff;border:1px solid #d9e9f8;border-radius:13px;padding:11px 12px;height:82px;box-shadow:0 4px 12px rgba(28,88,140,.04)}
.kpi-label{font-size:10px;color:#7b90a7;text-transform:uppercase;letter-spacing:.5px}.kpi-value{font-size:19px;font-weight:850;color:#173f69;margin-top:5px}.kpi-sub{font-size:10px;color:#7b90a7}
.badge{display:inline-block;padding:4px 9px;border-radius:999px;background:#eaf5ff;color:#1677c9;font-size:10px;font-weight:700;margin:3px 4px 0 0}.badge-green{background:#e7fbf3;color:#078a63}.badge-red{background:#fff0f2;color:#d04453}.badge-gold{background:#fff7e7;color:#b6770a}
.score {font-size:35px;font-weight:950;color:#1687f8;line-height:1}.score-label{font-size:12px;font-weight:800;color:#0c9870}
.news-card{background:#fff;border:1px solid #dcebf8;border-radius:13px;padding:10px 11px;margin-bottom:8px}.news-title{font-size:12px;font-weight:750;color:#234c75;line-height:1.35}.news-meta{font-size:9px;color:#8195a9;margin-top:4px}.news-sent{float:right;font-size:9px;font-weight:800;padding:2px 6px;border-radius:8px}.sent-pos{background:#e8faf3;color:#078b63}.sent-neg{background:#fff0f2;color:#d14453}.sent-neu{background:#eef3f7;color:#667b90}
.stock-button button{width:100%;text-align:left;border:1px solid #d8e8f6!important;background:#fff!important;color:#174873!important;border-radius:9px!important;font-weight:800!important;padding:7px 9px!important;}
.stock-button button:hover{border-color:#1687f8!important;background:#f1f8ff!important;color:#0b6fc3!important;}
.stButton>button{border-radius:10px;border:1px solid #cfe4f6;background:#fff;color:#164774;font-weight:750;min-height:38px;transition:all .15s ease}.stButton>button:hover{border-color:#1687f8;color:#0875c8;background:#f2f9ff}
div[data-testid="stMetric"]{background:#fff;border:1px solid #d9e9f8;border-radius:12px;padding:8px 11px;box-shadow:0 3px 10px rgba(20,85,140,.04)}
div[data-testid="stMetricLabel"]{font-size:10px!important;line-height:1.2!important;font-weight:700!important;color:#71879c!important}
div[data-testid="stMetricValue"]{font-size:16px!important;line-height:1.15!important;font-weight:750!important;color:#173f69!important;white-space:nowrap!important}
div[data-testid="stMetricDelta"]{font-size:10px!important}
.stMarkdown p{font-size:13px;line-height:1.45}
.stMarkdown h1{font-size:24px!important}.stMarkdown h2{font-size:20px!important}.stMarkdown h3{font-size:16px!important}.stMarkdown h4{font-size:14px!important}
.stCaption,.stCaption p{font-size:10px!important}
.market-card-price{font-size:15px;font-weight:800;color:#173f69;line-height:1.1;margin-top:2px}.market-card-change{font-size:10px;font-weight:750;margin-top:2px}
.stTabs [data-baseweb="tab-list"]{gap:3px;background:#fff;border:1px solid #d9e9f8;border-radius:12px;padding:4px;margin-bottom:10px;box-shadow:0 3px 10px rgba(20,85,140,.04)}
.stTabs [data-baseweb="tab"]{height:40px;border-radius:9px;padding:0 13px;color:#55728f;font-size:11px;font-weight:750;transition:all .15s ease}.stTabs [data-baseweb="tab"]:hover{background:#f1f8ff;color:#0876ca}.stTabs [aria-selected="true"]{background:#e9f5ff;color:#0876ca}
div[data-baseweb="select"]>div{border-radius:9px!important;border-color:#cfe3f5!important;background:#fff!important}.stTextInput input{border-radius:9px!important;border-color:#cfe3f5!important;background:#fff!important}
.section-rule{height:1px;background:#e2edf6;margin:8px 0 12px}.footer{font-size:10px;color:#8296aa;text-align:center;padding:14px}
.calendar-cell{background:#fff;border:1px solid #dcebf7;border-radius:12px;padding:8px;min-height:116px;box-shadow:0 3px 10px rgba(22,83,133,.035)}
.calendar-date{font-size:11px;font-weight:900;color:#174a77;margin-bottom:5px}.calendar-event{font-size:10px;margin:3px 0;padding:4px 5px;border-radius:6px;background:#eef7ff;color:#1c5e91}
.intelligence-hero{background:linear-gradient(120deg,#ffffff 0%,#eef8ff 55%,#e8f4ff 100%);border:1px solid #cfe5f8;border-radius:20px;padding:18px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;box-shadow:0 10px 28px rgba(22,95,150,.08)}.ai-live{font-size:10px;font-weight:900;letter-spacing:.7px;color:#0a9b72;background:#e8faf3;border:1px solid #c7efdf;border-radius:999px;padding:7px 10px}.rank-card{background:#fff;border:1px solid #d9e9f8;border-radius:14px;padding:12px 14px;margin-bottom:7px;box-shadow:0 4px 14px rgba(28,88,140,.045)}.rank-top{display:flex;justify-content:space-between;color:#173f69;font-size:15px}.rank-score{color:#1687f8;font-weight:900}.rank-reason{font-size:11px;color:#3e607f;margin-top:5px}.rank-meta{font-size:9px;color:#7d93a8;margin-top:6px}.scanner-banner{background:linear-gradient(120deg,#fff,#f1f8ff);border:1px solid #cfe5f8;border-radius:16px;padding:13px 16px;margin-bottom:10px}.scanner-mini{font-size:9px;text-transform:uppercase;letter-spacing:.5px;color:#7d93a8;font-weight:800;margin-bottom:4px}.small-table{font-size:11px}.hint{background:#f4faff;border:1px dashed #c7e1f6;border-radius:10px;padding:8px 10px;font-size:11px;color:#64809a}
/* Standard, responsive typography — avoid viewport-scaled oversized text */
html, body, [class*="css"] { overflow-wrap:anywhere; }
.stApp, .stApp * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }
.stApp { font-size:14px; }
.stMarkdown, .stText, .stCaption, .stAlert, .stButton, .stSelectbox, .stTextInput, .stCheckbox, .stToggle, .stSlider { font-size:14px; }
.stMarkdown p, .stMarkdown li { font-size:14px; line-height:1.5; }
.stMarkdown h1 { font-size:1.65rem !important; line-height:1.2 !important; }
.stMarkdown h2 { font-size:1.35rem !important; line-height:1.25 !important; }
.stMarkdown h3 { font-size:1.15rem !important; line-height:1.3 !important; }
.stMarkdown h4 { font-size:1rem !important; line-height:1.35 !important; }
.brand { font-size:20px; }
.brand small { font-size:11px; line-height:1.35; }
.side-brand-title { font-size:15px; }
.side-brand-sub, .card-sub, .hint, .news-meta { font-size:11px; }
.card-title { font-size:14px; }
.hero-ticker { font-size:24px; }
.hero-company { font-size:12px; }
.hero-price { font-size:27px; }
.market-value { font-size:16px; }
.market-change { font-size:11px; }
.market-name { font-size:10px; }
.kpi-label { font-size:10px; }
.kpi-value { font-size:17px; }
.kpi-sub { font-size:10px; }
.rank-top { font-size:14px; }
.rank-reason { font-size:11px; }
.rank-meta { font-size:10px; }
.forecast-title { font-size:13px; }
.forecast-sub { font-size:11px; }
.edu-term { font-size:12px; }
.edu-def { font-size:11px; }
.news-title { font-size:12px; }
.stButton > button { white-space:normal !important; overflow-wrap:anywhere !important; font-size:13px !important; min-height:36px !important; }
.stTabs [data-baseweb="tab-list"] { overflow-x:auto; scrollbar-width:thin; }
.stTabs [data-baseweb="tab"] { font-size:12px; padding:0 11px; white-space:nowrap; }
[data-testid="stDataFrame"] { max-width:100%; }
@media (max-width:1100px) {
  .block-container { padding-left:.65rem; padding-right:.65rem; }
  .market-value { font-size:15px; }
  .hero-price { font-size:25px; }
}
@media (max-width:800px) {
  .brand-wrap { min-height:46px; }
  .logo-large { width:40px; height:40px; }
  .hero-card { padding:13px; }
  .hero-ticker { font-size:21px; }
  .hero-price { font-size:23px; }
  .stMarkdown p, .stMarkdown li { font-size:13px; }
  .stTabs [data-baseweb="tab"] { font-size:11px; padding:0 9px; }
}
/* Market cards: one visual field for ticker + price + change + graph */
.market-card-wrap { background:#fff; border:1px solid #d9e9f8; border-radius:14px; padding:8px 10px 6px; box-shadow:0 4px 14px rgba(28,88,140,.045); height:100%; }
.market-card-label { font-size:10px; color:#6f849b; font-weight:800; text-transform:uppercase; letter-spacing:.35px; margin-bottom:4px; }
.market-card-price { font-size:16px; font-weight:850; color:#183e67; }
.market-card-change { font-size:11px; font-weight:800; margin-top:1px; }
.market-card-wrap .stButton { margin:0 !important; }
.market-card-wrap .stButton > button { border:0 !important; background:transparent !important; box-shadow:none !important; padding:0 !important; min-height:25px !important; justify-content:flex-start !important; color:#0b6fc3 !important; font-size:12px !important; font-weight:850 !important; }
.market-card-wrap .stButton > button:hover { background:#f1f8ff !important; border-radius:7px !important; }


</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Utilities
# ============================================================
def safe_float(v):
    try:
        if v is None or v == "": return np.nan
        return float(v)
    except Exception:
        return np.nan


def fmt_money(v):
    x = safe_float(v)
    if not np.isfinite(x): return "—"
    if abs(x) >= 1e12: return f"${x/1e12:.2f}T"
    if abs(x) >= 1e9: return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6: return f"${x/1e6:.2f}M"
    return f"${x:,.2f}"


def fmt_pct(v, digits=2):
    x = safe_float(v)
    return "—" if not np.isfinite(x) else f"{x:.{digits}f}%"


def fmt_num(v, digits=2):
    x = safe_float(v)
    return "—" if not np.isfinite(x) else f"{x:.{digits}f}"


def select_symbol(ticker: str, action: str = "Selected", open_details: bool = False):
    """Update the active symbol before Streamlit reruns the script."""
    t = str(ticker).upper().strip()
    if t:
        st.session_state.selected_symbol = t
        # Keep the sidebar search widget synchronized with every clickable ticker.
        st.session_state.sidebar_stock_search = t
        if open_details:
            st.session_state.stock_detail_open = True
        log_stock_event(t, action)

def open_stock_details(ticker: str, action: str = "Stock details"):
    select_symbol(ticker, action, open_details=True)


def set_symbol(ticker: str):
    """Immediate selection helper for non-callback contexts."""
    select_symbol(ticker)
    st.rerun()


def clickable_ticker(ticker: str, key: str, label: str = "", action: str = "Selected"):
    text = ticker if not label else f"{ticker}  ·  {label}"
    st.button(text, key=key, use_container_width=True, on_click=open_stock_details, args=(ticker, action))

# ============================================================
# Data access
# ============================================================
@st.cache_data(ttl=90, show_spinner=False)
def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        # Yahoo intraday retention limits: keep requested periods useful instead of returning empty data.
        effective_period = period
        if interval in {"5m", "15m", "30m"}:
            effective_period = "60d" if period in {"3mo","6mo","1y","2y","5y"} else period
        elif interval == "1h" and period in {"2y","5y"}:
            effective_period = "730d"
        df = yf.download(ticker, period=effective_period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=180, show_spinner=False)
def get_info(ticker: str) -> Dict[str, Any]:
    try: return yf.Ticker(ticker).info or {}
    except Exception: return {}


@st.cache_data(ttl=120, show_spinner=False)
def get_news(ticker: str, limit: int = 15) -> List[Dict[str, Any]]:
    try: raw = yf.Ticker(ticker).news or []
    except Exception: raw = []
    out=[]
    for item in raw[:limit]:
        c=item.get("content",item)
        title=c.get("title") or item.get("title") or ""
        pub=c.get("pubDate") or item.get("providerPublishTime")
        click=c.get("clickThroughUrl") or {}
        url=click.get("url","") if isinstance(click,dict) else ""
        url=url or item.get("link","")
        provider=c.get("provider",{})
        publisher=provider.get("displayName","Yahoo Finance") if isinstance(provider,dict) else str(provider)
        if title: out.append({"title":title,"url":url,"publisher":publisher,"published":pub,"ticker":ticker})
    return out


def benzinga_get(endpoint: str, params: Dict[str, Any]) -> Any:
    if not BENZINGA_KEY: return None
    try:
        p=dict(params); p["token"]=BENZINGA_KEY
        r=requests.get(f"{BENZINGA_BASE}/{endpoint.lstrip('/')}",params=p,timeout=10,headers={"User-Agent":"AIStockAnalyzerPro/5.0","Accept":"application/json"})
        r.raise_for_status(); return r.json()
    except Exception: return None


@st.cache_data(ttl=120, show_spinner=False)
def benzinga_news(ticker: str = "") -> List[Dict[str, Any]]:
    if not BENZINGA_KEY: return []
    p={"pageSize":25,"sort":"-created"}
    if ticker: p["tickers"]=ticker
    data=benzinga_get("news",p)
    rows=(data or {}).get("data",[]) if isinstance(data,dict) else []
    return [{"title":x.get("title",""),"url":x.get("url",""),"publisher":"Benzinga","published":x.get("created"),"ticker":ticker} for x in rows if x.get("title")]


@st.cache_data(ttl=120, show_spinner=False)
def market_news() -> List[Dict[str, Any]]:
    seen=set(); out=[]
    if BENZINGA_KEY:
        out.extend(benzinga_news(""))
    for t in ["SPY","QQQ","NVDA","AAPL","MSFT","AMZN","META","TSLA","AMD","AVGO","ORCL"]:
        for n in get_news(t,5):
            k=n["title"].lower().strip()
            if k in seen: continue
            seen.add(k); out.append(n)
    return out[:40]


def combined_news(ticker: str, limit=15):
    b=benzinga_news(ticker) if BENZINGA_KEY else []
    return (b or get_news(ticker,limit))[:limit]

# ============================================================
# LLM intelligence layer
# ============================================================
def llm_available(provider: str) -> bool:
    if provider == "OpenAI": return bool(OPENAI_KEY and OpenAI)
    if provider == "Claude": return bool(ANTHROPIC_KEY and Anthropic)
    if provider == "Local Ollama": return bool(OLLAMA_URL)
    if provider == "Auto": return bool((OPENAI_KEY and OpenAI) or (ANTHROPIC_KEY and Anthropic))
    return False

def llm_provider_default() -> str:
    if OPENAI_KEY and OpenAI: return "OpenAI"
    if ANTHROPIC_KEY and Anthropic: return "Claude"
    return "Disabled"

def _llm_prompt(report_rows: pd.DataFrame, news_map: Dict[str, List[Dict[str,Any]]], regime_label: str) -> str:
    rows=[]
    for _,r in report_rows.iterrows():
        t=str(r.get("Ticker","")); headlines=[n.get("title","") for n in news_map.get(t,[])[:5] if n.get("title")]
        rows.append({
            "ticker":t,"decision":r.get("Decision"),"score":int(r.get("Score",0)),
            "price":safe_float(r.get("Price")),"technical":int(r.get("Technical",50)),
            "fundamental":int(r.get("Fundamental",50)),"earnings":int(r.get("Earnings",50)),
            "news":int(r.get("News",50)),"volume":int(r.get("Volume",50)),"options":int(r.get("Options",50)),
            "market":int(r.get("Market",50)),"momentum":safe_float(r.get("Momentum")),
            "relative_volume":safe_float(r.get("RelVol")),"rsi":safe_float(r.get("RSI")),
            "earnings_signal":r.get("Earnings Signal"),"options_signal":r.get("Options Signal"),
            "headlines":headlines
        })
    return f"""You are the senior market-research editor for a stock dashboard.
The quantitative engine has already screened hundreds of liquid US symbols. Do NOT invent facts, targets, catalysts, earnings dates, analyst opinions, or news. Use only the supplied data. Clearly say when data is unavailable.
Market regime: {regime_label}.

For each of the 10 stocks, explain WHY the quantitative decision is BUY, WATCH, or AVOID in 2-4 concise sentences. Compare the strongest catalysts across the list, identify which catalysts are confirmed by the supplied headlines, and flag risks such as elevated RSI, weak fundamentals, put-heavy options, weak market regime, or low/negative momentum.

Finish with:
1) Top 3 opportunities and why they stand out
2) Top 3 catalysts to watch today
3) Key risks / invalidation signals
4) A concise portfolio-style conclusion (research only, not personalized investment advice)

Return clean Markdown with a numbered Top 10 list.

DATA:
{json.dumps(rows, default=str, indent=2)}"""

def generate_llm_report(report_rows: pd.DataFrame, news_map: Dict[str,List[Dict[str,Any]]], regime_label: str, provider: str, model: str) -> str:
    prompt=_llm_prompt(report_rows,news_map,regime_label)
    try:
        if provider == "OpenAI" and OPENAI_KEY and OpenAI:
            client=OpenAI(api_key=OPENAI_KEY)
            response=client.responses.create(model=model or OPENAI_MODEL_DEFAULT, input=prompt)
            return response.output_text.strip()
        if provider == "Claude" and ANTHROPIC_KEY and Anthropic:
            client=Anthropic(api_key=ANTHROPIC_KEY)
            msg=client.messages.create(model=model or CLAUDE_MODEL_DEFAULT, max_tokens=5000, messages=[{"role":"user","content":prompt}])
            return "\n".join([getattr(x,"text",str(x)) for x in msg.content]).strip()
        if provider == "Local Ollama":
            r=requests.post(OLLAMA_URL,json={"model":model or "llama3.2:3b","prompt":prompt,"stream":False},timeout=120)
            r.raise_for_status(); return str(r.json().get("response","")).strip()
        if provider == "Auto":
            p=llm_provider_default()
            return generate_llm_report(report_rows,news_map,regime_label,p,model)
    except Exception as e:
        logger.exception("LLM report generation failed")
        return f"LLM report generation failed: {e}"
    return "LLM is not configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, or use Local Ollama."

def daily_report_path() -> Path:
    return REPORT_DIR / f"stock_report_{date.today().isoformat()}.md"

def save_daily_report(text: str) -> Path:
    path=daily_report_path()
    path.write_text(f"# AI Stock Analyzer Pro — Daily Top 10\n\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n\n{text}\n",encoding="utf-8")
    return path

@st.cache_data(ttl=86400, show_spinner=False)
def cached_llm_report(report_key: str, provider: str, model: str, report_json: str, news_json: str, regime_label: str) -> str:
    rows=pd.DataFrame(json.loads(report_json))
    news_map=json.loads(news_json)
    return generate_llm_report(rows,news_map,regime_label,provider,model)

# ============================================================
# Indicators / scoring
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    d=df.copy(); close=d["Close"].astype(float); high=d["High"].astype(float); low=d["Low"].astype(float); vol=d["Volume"].astype(float)
    for n in [9,21,50,200]: d[f"MA{n}"]=close.rolling(n).mean()
    delta=close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); d["RSI"]=100-(100/(1+rs))
    ema12=close.ewm(span=12,adjust=False).mean(); ema26=close.ewm(span=26,adjust=False).mean()
    d["MACD"]=ema12-ema26; d["MACDSignal"]=d["MACD"].ewm(span=9,adjust=False).mean(); d["MACDHist"]=d["MACD"]-d["MACDSignal"]
    d["VolAvg20"]=vol.rolling(20).mean(); d["RelVol"]=vol/d["VolAvg20"]
    tr=pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1); d["ATR14"]=tr.rolling(14).mean()
    d["BBMid"]=close.rolling(20).mean(); sd=close.rolling(20).std(); d["BBUpper"]=d["BBMid"]+2*sd; d["BBLower"]=d["BBMid"]-2*sd
    d["Ret1D"]=close.pct_change()*100; d["Ret5D"]=close.pct_change(5)*100; d["Ret20D"]=close.pct_change(20)*100
    return d


def classify_sentiment(text: str) -> Tuple[str,int]:
    t=(text or "").lower()
    pos=["beat","beats","upgrade","upgrades","buy","bullish","surge","rally","record","strong","growth","raised","approval","contract","deal","launch","outperform","ai demand"]
    neg=["miss","misses","downgrade","downgrades","sell","bearish","drop","falls","fell","weak","cut","cuts","warning","probe","lawsuit","offering","tariff","inflation","recall"]
    p=sum(t.count(w) for w in pos); n=sum(t.count(w) for w in neg); score=max(-100,min(100,(p-n)*18))
    return ("Positive" if score>10 else "Negative" if score<-10 else "Neutral"),score


def score_stock(df: pd.DataFrame, info: Dict[str,Any], news: List[Dict[str,Any]]) -> Dict[str,Any]:
    if df.empty: return {"score":0,"label":"No data","news_score":0}
    d=add_indicators(df); x=d.iloc[-1]; price=safe_float(x["Close"]); score=50; notes=[]
    for ma,pts in [("MA21",6),("MA50",8),("MA200",10)]:
        v=safe_float(x[ma])
        if np.isfinite(v):
            if price>v: score+=pts; notes.append(f"Above {ma}")
            else: score-=pts; notes.append(f"Below {ma}")
    rsi=safe_float(x["RSI"])
    if np.isfinite(rsi):
        if 50<=rsi<=70: score+=8
        elif rsi>75: score-=4
        elif rsi<30: score+=2
        elif rsi<45: score-=5
    if safe_float(x["MACD"])>safe_float(x["MACDSignal"]): score+=7
    else: score-=7
    rv=safe_float(x["RelVol"])
    if np.isfinite(rv) and rv>=1.5: score+=6
    ns=int(np.mean([classify_sentiment(n.get("title",""))[1] for n in news[:10]])) if news else 0
    score+=max(-8,min(8,ns/10)); score=int(max(0,min(100,round(score))))
    label="Bullish" if score>=72 else "Constructive" if score>=58 else "Neutral" if score>=45 else "Cautious" if score>=32 else "Bearish"
    momentum=safe_float(x["Ret20D"])
    return {"score":score,"label":label,"price":price,"rsi":rsi,"relvol":rv,"news_score":ns,"notes":notes,"momentum":momentum}


def market_regime() -> Dict[str,Any]:
    rows=[]
    for t in ["SPY","QQQ","IWM","TLT","GLD","USO"]:
        df=get_history(t,"6mo","1d")
        if df.empty: continue
        d=add_indicators(df); x=d.iloc[-1]
        rows.append({"Ticker":t,"Price":safe_float(x["Close"]),"1D %":safe_float(x["Ret1D"]),"20D %":safe_float(x["Ret20D"]),"MA50":safe_float(x["MA50"])})
    reg=pd.DataFrame(rows)
    if reg.empty: return {"label":"Unknown","score":50,"table":reg}
    risk=float(reg.loc[reg.Ticker.isin(["SPY","QQQ","IWM"]),"20D %"].mean())
    label="Risk-On" if risk>2 else "Neutral" if risk>-2 else "Risk-Off"
    return {"label":label,"score":int(max(0,min(100,50+risk*4))),"table":reg}

# ============================================================
# Movers / earnings / analysts / options / IPO
# ============================================================
@st.cache_data(ttl=120, show_spinner=False)
def get_screener(kind: str, count: int=20) -> pd.DataFrame:
    try:
        url="https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        r=requests.get(url,params={"scrIds":kind,"count":count},timeout=10,headers={"User-Agent":"Mozilla/5.0"}); r.raise_for_status()
        quotes=r.json().get("finance",{}).get("result",[{}])[0].get("quotes",[])
        return pd.DataFrame(quotes)
    except Exception: return pd.DataFrame()


def safe_market_cap_filter(df: pd.DataFrame, minimum: str) -> pd.DataFrame:
    cap_map={"Any":0,"$2B+":2e9,"$10B+":10e9,"$50B+":50e9}
    threshold=cap_map.get(minimum,0)
    if df.empty or "marketCap" not in df.columns: return df
    return df[pd.to_numeric(df["marketCap"],errors="coerce").fillna(0)>=threshold].copy()


def mover_reason(ticker: str) -> Tuple[str,str]:
    ns=get_news(ticker,5)
    if not ns: return "No clear headline","No recent headline was returned by the available feed."
    title=ns[0].get("title",""); low=title.lower()
    mapping=[
        (("earnings","eps","revenue","profit","results","guidance"),"Earnings / Guidance"),
        (("upgrade","downgrade","price target","analyst"),"Analyst action"),
        (("deal","contract","acquire","acquisition","merger","partnership"),"Deal / Contract"),
        (("launch","product","ai","chip","robotaxi"),"Product / AI"),
        (("fda","regulator","sec","lawsuit","probe","approval"),"Regulatory"),
        (("offering","shares","dilution","financing"),"Capital / Offering"),
        (("fed","rates","inflation","oil","tariff","treasury"),"Macro"),
    ]
    reason="News catalyst"
    for words,label in mapping:
        if any(w in low for w in words): reason=label; break
    return reason,title


def earnings_benzinga(days:int=30,ticker:str="") -> pd.DataFrame:
    if not BENZINGA_KEY: return pd.DataFrame()
    start=date.today(); end=start+timedelta(days=days)
    p={"date_from":str(start),"date_to":str(end),"pagesize":200}
    if ticker: p["tickers"]=ticker
    data=benzinga_get("calendar/earnings",p)
    rows=(data or {}).get("earnings",[]) if isinstance(data,dict) else []
    if not rows: rows=(data or {}).get("data",[]) if isinstance(data,dict) else []
    return pd.DataFrame(rows)


def earnings_yahoo(tickers:List[str]) -> pd.DataFrame:
    rows=[]
    for t in tickers:
        try:
            cal=yf.Ticker(t).calendar
            if isinstance(cal,pd.DataFrame): cal=cal.to_dict()
            if isinstance(cal,dict):
                dates=cal.get("Earnings Date",[]); dates=dates if isinstance(dates,list) else [dates]
                for dt in dates[:1]: rows.append({"Ticker":t,"Earnings Date":dt})
        except Exception: pass
    return pd.DataFrame(rows)


def earnings_events(days:int=30) -> pd.DataFrame:
    ed=earnings_benzinga(days)
    if not ed.empty: return ed
    return earnings_yahoo(WATCHLIST)


def normalize_earnings(df:pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    x=df.copy()
    ren={}
    for c in x.columns:
        lc=str(c).lower().replace("_"," ")
        if lc in ["ticker","symbol"]: ren[c]="Ticker"
        elif "date" in lc and "announce" not in lc: ren[c]="Earnings Date"
        elif "eps" in lc and ("estimate" in lc or "consensus" in lc): ren[c]="EPS Est"
        elif lc in ["eps actual","actual eps"]: ren[c]="EPS Actual"
        elif "time" in lc: ren[c]="Time"
    x=x.rename(columns=ren)
    if "Ticker" not in x.columns and "symbol" in x.columns: x=x.rename(columns={"symbol":"Ticker"})
    return x


def analyst_benzinga(ticker:str)->pd.DataFrame:
    if not BENZINGA_KEY: return pd.DataFrame()
    data=benzinga_get("calendar/ratings",{"tickers":ticker,"pagesize":50})
    rows=(data or {}).get("ratings",[]) if isinstance(data,dict) else []
    if not rows: rows=(data or {}).get("data",[]) if isinstance(data,dict) else []
    return pd.DataFrame(rows)


def options_data(ticker:str)->Tuple[pd.DataFrame,pd.DataFrame,Dict[str,Any]]:
    try:
        tk=yf.Ticker(ticker); dates=tk.options
        if not dates: return pd.DataFrame(),pd.DataFrame(),{}
        expiry=dates[0]; chain=tk.option_chain(expiry); calls=chain.calls.copy(); puts=chain.puts.copy()
        coi=float(calls["openInterest"].fillna(0).sum()); poi=float(puts["openInterest"].fillna(0).sum())
        cv=float(calls["volume"].fillna(0).sum()); pv=float(puts["volume"].fillna(0).sum())
        return calls,puts,{"expiry":expiry,"put_call_oi":poi/coi if coi else np.nan,"put_call_volume":pv/cv if cv else np.nan}
    except Exception: return pd.DataFrame(),pd.DataFrame(),{}

# ============================================================
# AI Market Scanner — free yfinance-first engine
# ============================================================
def _latest_scalar(series):
    try:
        v = series.iloc[-1]
        return safe_float(v)
    except Exception:
        return np.nan


def _technical_score_from_history(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or len(df) < 60:
        return {"technical":50,"price":np.nan,"momentum":np.nan,"relvol":np.nan,"rsi":np.nan,"macd":False}
    d=add_indicators(df.dropna(how="all")); x=d.iloc[-1]; price=safe_float(x.get("Close")); s=50
    for ma,pts in [("MA21",5),("MA50",7),("MA200",7)]:
        m=safe_float(x.get(ma));
        if np.isfinite(m): s += pts if price>m else -pts
    rsi=safe_float(x.get("RSI"))
    if np.isfinite(rsi):
        if 52<=rsi<=68: s+=6
        elif 45<=rsi<52: s+=1
        elif 68<rsi<=75: s+=2
        elif rsi>78: s-=4
        elif rsi<30: s+=2
        elif rsi<42: s-=5
    macd=safe_float(x.get("MACD")); sig=safe_float(x.get("MACDSignal")); macd_bull=bool(np.isfinite(macd) and np.isfinite(sig) and macd>sig)
    s += 5 if macd_bull else -5
    rv=safe_float(x.get("RelVol"));
    if np.isfinite(rv):
        if rv>=2: s+=5
        elif rv>=1.25: s+=3
    mom=safe_float(x.get("Ret20D"));
    if np.isfinite(mom): s += 5 if mom>5 else 3 if mom>2 else -4 if mom<-5 else 0
    return {"technical":int(max(0,min(100,s))),"price":price,"momentum":mom,"relvol":rv,"rsi":rsi,"macd":macd_bull}


def _fundamental_score(info: Dict[str,Any]) -> int:
    s=50
    pe=safe_float(info.get("forwardPE")); growth=safe_float(info.get("revenueGrowth")); margin=safe_float(info.get("profitMargins")); roe=safe_float(info.get("returnOnEquity")); cap=safe_float(info.get("marketCap"))
    if np.isfinite(pe): s += 8 if 0<pe<25 else 4 if pe<40 else -5 if pe>70 else 0
    if np.isfinite(growth): s += 10 if growth>0.20 else 6 if growth>0.08 else -5 if growth<0 else 0
    if np.isfinite(margin): s += 6 if margin>0.15 else 2 if margin>0 else -4
    if np.isfinite(roe): s += 5 if roe>0.15 else 2 if roe>0 else -3
    if np.isfinite(cap): s += 3 if cap>=10e9 else 0
    return int(max(0,min(100,s)))


def _news_score_for_scanner(news: List[Dict[str,Any]]) -> int:
    if not news: return 50
    vals=[classify_sentiment(n.get("title",""))[1] for n in news[:8]]
    return int(max(0,min(100,50+np.mean(vals)/2)))


def _earnings_score_for_scanner(ticker: str, info: Dict[str,Any]) -> Tuple[int,str]:
    score=50; note="No immediate earnings signal"
    try:
        cal=yf.Ticker(ticker).calendar
        if isinstance(cal,pd.DataFrame): cal=cal.to_dict()
        dates=cal.get("Earnings Date",[]) if isinstance(cal,dict) else []
        if not isinstance(dates,list): dates=[dates]
        if dates:
            dt=pd.to_datetime(dates[0],errors="coerce")
            if pd.notna(dt):
                days=(dt.date()-date.today()).days
                if 0<=days<=14: score+=12; note=f"Earnings in {days}d"
                elif 15<=days<=45: score+=5; note=f"Earnings in {days}d"
                elif days<0: score+=2; note="Recent earnings"
    except Exception: pass
    try:
        hist=yf.Ticker(ticker).earnings_history
        if isinstance(hist,pd.DataFrame) and not hist.empty:
            cols={str(c).lower():c for c in hist.columns}
            surprise=next((hist[c] for k,c in cols.items() if "surprise" in k),None)
            if surprise is not None:
                sv=pd.to_numeric(surprise,errors="coerce").tail(4).mean()
                if pd.notna(sv): score += 8 if sv>0.05 else 4 if sv>0 else -6 if sv<0 else 0
    except Exception: pass
    return int(max(0,min(100,score))),note


def _options_score_for_scanner(ticker: str) -> Tuple[int,str]:
    try:
        tk=yf.Ticker(ticker); dates=tk.options
        if not dates: return 50,"No options chain"
        ch=tk.option_chain(dates[0]); c=ch.calls; p=ch.puts
        coi=pd.to_numeric(c.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0).sum(); poi=pd.to_numeric(p.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()
        if coi<=0: return 50,"No call OI"
        ratio=poi/coi
        return (65,"Call-heavy") if ratio<0.75 else (55,"Balanced") if ratio<=1.25 else (35,"Put-heavy")
    except Exception:
        return 50,"Options unavailable"


def _market_score(regime: Dict[str,Any]) -> int:
    return int(max(0,min(100,regime.get("score",50))))


def _decision(score: int) -> str:
    if score>=75: return "BUY"
    if score>=55: return "WATCH"
    return "AVOID"


def _scanner_reason(row: Dict[str,Any]) -> str:
    parts=[]
    if row.get("technical",50)>=70: parts.append("strong technicals")
    if row.get("momentum",0)>5: parts.append("positive momentum")
    if row.get("relvol",0)>=1.5: parts.append("unusual volume")
    if row.get("fundamental",50)>=65: parts.append("healthy fundamentals")
    if row.get("news",50)>=60: parts.append("positive news")
    if row.get("earnings",50)>=60: parts.append("earnings catalyst")
    if row.get("options",50)>=60: parts.append("call-heavy options")
    return ", ".join(parts[:3]) or "mixed signals; monitor confirmation"


@st.cache_data(ttl=900, show_spinner=False)
def scanner_price_stage(universe: Tuple[str,...], period: str="6mo") -> pd.DataFrame:
    tickers=list(dict.fromkeys([t.upper().strip() for t in universe if t]))
    if not tickers: return pd.DataFrame()
    try:
        raw=yf.download(tickers=tickers,period=period,interval="1d",auto_adjust=False,progress=False,threads=True,group_by="column")
    except Exception:
        return pd.DataFrame()
    rows=[]
    if isinstance(raw.columns,pd.MultiIndex):
        level0=list(raw.columns.get_level_values(0).unique())
        if "Close" not in level0: return pd.DataFrame()
        for t in tickers:
            try:
                close=raw["Close"][t].dropna(); vol=raw["Volume"][t].dropna() if "Volume" in raw else pd.Series(index=close.index,dtype=float)
                if len(close)<60: continue
                d=pd.DataFrame({"Close":close,"Volume":vol}).dropna(subset=["Close"]); ti=_technical_score_from_history(d)
                rows.append({"Ticker":t,**ti})
            except Exception: continue
    else:
        ti=_technical_score_from_history(raw)
        if ti.get("price")==ti.get("price"): rows.append({"Ticker":tickers[0],**ti})
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def run_ai_market_scan(universe: Tuple[str,...], enrich_count: int, regime_label: str, regime_score: int) -> pd.DataFrame:
    base=scanner_price_stage(universe,"6mo")
    if base.empty: return base
    base=base.sort_values(["technical","momentum"],ascending=False).reset_index(drop=True)
    # Enrich only the strongest candidates to keep a free yfinance scan responsive.
    candidates=base.head(max(20,min(enrich_count,len(base)))).copy()
    rows=[]
    for _,r in candidates.iterrows():
        t=r["Ticker"]; info=get_info(t); news=get_news(t,6)
        f=_fundamental_score(info); n=_news_score_for_scanner(news); e,enote=_earnings_score_for_scanner(t,info); o,onote=_options_score_for_scanner(t)
        tech=int(r["technical"]); vol=50
        rv=safe_float(r.get("relvol"));
        if np.isfinite(rv): vol=70 if rv>=2 else 60 if rv>=1.5 else 50 if rv>=1 else 40
        market=_market_score({"score":regime_score,"label":regime_label})
        # Weighted model: technical 25, fundamentals 20, earnings 15, news 15, unusual volume 10, options 5, regime 10.
        composite=round(tech*.25+f*.20+e*.15+n*.15+vol*.10+o*.05+market*.10)
        decision=_decision(composite)
        rows.append({"Ticker":t,"Decision":decision,"Score":composite,"Technical":tech,"Fundamental":f,"Earnings":e,"News":n,"Volume":vol,"Options":o,"Market":market,"Price":r.get("price"),"Momentum":r.get("momentum"),"RelVol":r.get("relvol"),"RSI":r.get("rsi"),"Earnings Signal":enote,"Options Signal":onote,"Reason":_scanner_reason({"technical":tech,"fundamental":f,"earnings":e,"news":n,"options":o,"momentum":r.get("momentum"),"relvol":r.get("relvol")}),"Market Regime":regime_label})
    out=pd.DataFrame(rows)
    if out.empty: return out
    return out.sort_values(["Decision","Score"],key=lambda s:s.map({"BUY":0,"WATCH":1,"AVOID":2}) if s.name=="Decision" else s,ascending=[True,False]).reset_index(drop=True)


# ============================================================
# AI Daily Intelligence Center / history / market breadth
# ============================================================
def _read_recommendation_history() -> pd.DataFrame:
    try:
        if HISTORY_FILE.exists():
            return pd.read_csv(HISTORY_FILE)
    except Exception:
        logger.exception("Unable to read recommendation history")
    return pd.DataFrame(columns=["Date","Ticker","Decision","Score","Price","Momentum","RelVol","Reason"])


def _write_recommendation_history(df: pd.DataFrame):
    if df.empty: return
    try:
        existing=_read_recommendation_history()
        cols=["Date","Ticker","Decision","Score","Price","Momentum","RelVol","Reason"]
        add=df.copy()
        add["Date"]=date.today().isoformat()
        for c in cols:
            if c not in add.columns: add[c]=np.nan
        add=add[cols]
        combined=pd.concat([existing,add],ignore_index=True)
        combined=combined.drop_duplicates(subset=["Date","Ticker"],keep="last")
        combined.to_csv(HISTORY_FILE,index=False)
    except Exception:
        logger.exception("Unable to save recommendation history")


@st.cache_data(ttl=300, show_spinner=False)
def historical_recommendation_performance() -> pd.DataFrame:
    hist=_read_recommendation_history()
    if hist.empty: return hist
    rows=[]
    for _,r in hist.tail(500).iterrows():
        t=str(r.get("Ticker","")).upper().strip()
        if not t: continue
        p0=safe_float(r.get("Price"))
        if not np.isfinite(p0) or p0<=0: continue
        h=get_history(t,"3mo","1d")
        if h.empty: continue
        close=pd.to_numeric(h["Close"],errors="coerce").dropna()
        if close.empty: continue
        rec_date=pd.to_datetime(r.get("Date"),errors="coerce")
        current=safe_float(close.iloc[-1])
        row={"Date":r.get("Date"),"Ticker":t,"Decision":r.get("Decision"),"Score":safe_float(r.get("Score")),"Entry":p0,"Current":current}
        row["Return %"]=(current/p0-1)*100
        try:
            future=close[close.index>=rec_date]
            # Current return is the authoritative result; 5D/20D are populated when enough observations exist.
            vals=list(future.values)
            row["5D %"]=(vals[5]/p0-1)*100 if len(vals)>5 else np.nan
            row["20D %"]=(vals[20]/p0-1)*100 if len(vals)>20 else np.nan
        except Exception:
            row["5D %"]=np.nan; row["20D %"]=np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def sector_rotation_snapshot() -> pd.DataFrame:
    sector_etfs={"Technology":"XLK","Financials":"XLF","Energy":"XLE","Healthcare":"XLV","Consumer Disc.":"XLY","Industrials":"XLI","Communication":"XLC","Utilities":"XLU","Staples":"XLP","Materials":"XLB","Real Estate":"XLRE"}
    rows=[]
    for sector,t in sector_etfs.items():
        h=get_history(t,"3mo","1d")
        if h.empty: continue
        d=add_indicators(h); x=d.iloc[-1]
        rows.append({"Sector":sector,"ETF":t,"1D %":safe_float(x.get("Ret1D")),"5D %":safe_float(x.get("Ret5D")),"20D %":safe_float(x.get("Ret20D")),"RSI":safe_float(x.get("RSI"))})
    out=pd.DataFrame(rows)
    return out.sort_values("20D %",ascending=False).reset_index(drop=True) if not out.empty else out


def unusual_options_snapshot(tickers: List[str], limit: int=12) -> pd.DataFrame:
    rows=[]
    for t in tickers[:limit]:
        try:
            calls,puts,meta=options_data(t)
            if not meta: continue
            cv=pd.to_numeric(calls.get("volume",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()
            pv=pd.to_numeric(puts.get("volume",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()
            coi=pd.to_numeric(calls.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()
            poi=pd.to_numeric(puts.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()
            total_oi=coi+poi; total_vol=cv+pv
            vol_oi=total_vol/total_oi if total_oi else np.nan
            pcv=pv/cv if cv else np.nan
            if np.isfinite(vol_oi) and vol_oi>=0.75:
                rows.append({"Ticker":t,"Expiry":meta.get("expiry"),"Option Vol":int(total_vol),"OI":int(total_oi),"Vol/OI":vol_oi,"Put/Call Vol":pcv,"Signal":"Unusual options activity"})
        except Exception: continue
    out=pd.DataFrame(rows)
    return out.sort_values("Vol/OI",ascending=False).head(10) if not out.empty else out


def analyst_upgrade_downgrade_snapshot(tickers: List[str]) -> pd.DataFrame:
    rows=[]
    if BENZINGA_KEY:
        for t in tickers[:20]:
            try:
                d=analyst_benzinga(t)
                if d.empty: continue
                for _,r in d.head(5).iterrows():
                    action=str(r.get("action_company",r.get("action","")))
                    low=action.lower()
                    if any(k in low for k in ["upgrade","downgrade","initiated","reiterate","maintain"]):
                        rows.append({"Ticker":t,"Date":r.get("date",""),"Firm":r.get("firm",r.get("analyst","")),"Action":action,"Rating":r.get("rating_current",r.get("rating","")),"Target":r.get("pt_current",np.nan)})
            except Exception: continue
    return pd.DataFrame(rows)


def news_catalyst_snapshot(tickers: List[str], limit: int=10) -> pd.DataFrame:
    rows=[]
    for t in tickers[:limit]:
        ns=combined_news(t,5)
        if not ns: continue
        title=ns[0].get("title","")
        reason,_=mover_reason(t)
        sent,_sc=classify_sentiment(title)
        rows.append({"Ticker":t,"Catalyst":reason,"Sentiment":sent,"Headline":title,"Publisher":ns[0].get("publisher","")})
    return pd.DataFrame(rows)


def quantitative_daily_briefing(scan: pd.DataFrame, regime: Dict[str,Any]) -> str:
    if scan.empty:
        return f"Market regime: {regime.get('label','Unknown')}. The scanner did not return enough data for today's ranking."
    buys=scan[scan.Decision=="BUY"].head(3)["Ticker"].tolist()
    watches=scan[scan.Decision=="WATCH"].head(3)["Ticker"].tolist()
    avoid=scan[scan.Decision=="AVOID"].head(3)["Ticker"].tolist()
    return (f"**Market regime:** {regime.get('label','Unknown')} ({regime.get('score',50)}/100). "
            f"**Quant leaders:** {', '.join(buys) or 'none'}. "
            f"**Watch:** {', '.join(watches) or 'none'}. "
            f"**Avoid / weak setups:** {', '.join(avoid) or 'none'}. "
            "LLM commentary is optional; rankings below are generated from the quantitative framework.")


def daily_intelligence_prompt(scan: pd.DataFrame, regime: Dict[str,Any], sector: pd.DataFrame, analyst: pd.DataFrame, news_df: pd.DataFrame) -> str:
    rows=scan.head(10).replace({np.nan:None}).to_dict("records")
    sector_rows=sector.head(6).replace({np.nan:None}).to_dict("records") if not sector.empty else []
    analyst_rows=analyst.head(10).replace({np.nan:None}).to_dict("records") if not analyst.empty else []
    news_rows=news_df.head(10).replace({np.nan:None}).to_dict("records") if not news_df.empty else []
    return f"""You are the chief market-intelligence editor. Produce a concise professional morning stock report using ONLY the supplied data. Never invent a catalyst, analyst target, earnings date, news fact, or performance result. If data is unavailable, say so.\n\nMarket regime: {regime.get('label')} ({regime.get('score')}/100).\n\nRequired sections:\n# Morning Market Briefing\n- Market regime and what it means for risk appetite\n- Sector rotation leaders/laggards\n- Important catalysts and unusual activity\n\n# AI Top 10 BUY / WATCH / AVOID\nFor every ranked stock explain: Why this stock today? What is the strongest confirming evidence? What could make the recommendation wrong?\n\n# Biggest Upgrades / Downgrades\nSummarize only supplied analyst actions.\n\n# Earnings / News Catalysts\nSummarize only supplied rows.\n\n# Risk & Invalidation Signals\n\n# Bottom Line\nKeep the report readable for a morning investor/research workflow. This is research, not personalized investment advice.\n\nTOP SCAN DATA:\n{json.dumps(rows,default=str,indent=2)}\n\nSECTOR DATA:\n{json.dumps(sector_rows,default=str,indent=2)}\n\nANALYST ACTIONS:\n{json.dumps(analyst_rows,default=str,indent=2)}\n\nNEWS CATALYSTS:\n{json.dumps(news_rows,default=str,indent=2)}"""


def generate_daily_intelligence(scan: pd.DataFrame, regime: Dict[str,Any], sector: pd.DataFrame, analyst: pd.DataFrame, news_df: pd.DataFrame, provider: str, model: str) -> str:
    prompt=daily_intelligence_prompt(scan,regime,sector,analyst,news_df)
    try:
        if provider == "OpenAI" and OPENAI_KEY and OpenAI:
            client=OpenAI(api_key=OPENAI_KEY)
            return client.responses.create(model=model or OPENAI_MODEL_DEFAULT,input=prompt).output_text.strip()
        if provider == "Claude" and ANTHROPIC_KEY and Anthropic:
            client=Anthropic(api_key=ANTHROPIC_KEY)
            msg=client.messages.create(model=model or CLAUDE_MODEL_DEFAULT,max_tokens=6000,messages=[{"role":"user","content":prompt}])
            return "\n".join([getattr(x,"text",str(x)) for x in msg.content]).strip()
        if provider == "Local Ollama":
            r=requests.post(OLLAMA_URL,json={"model":model or "llama3.2:3b","prompt":prompt,"stream":False},timeout=180); r.raise_for_status(); return str(r.json().get("response","")).strip()
    except Exception as e:
        logger.exception("Daily intelligence generation failed")
        return f"AI report generation failed: {e}\n\n{quantitative_daily_briefing(scan,regime)}"
    return quantitative_daily_briefing(scan,regime)


def save_intelligence_report(text: str) -> Path:
    path=REPORT_DIR / f"daily_intelligence_{date.today().isoformat()}.md"
    path.write_text(f"# AI Stock Analyzer Pro — Daily Intelligence\n\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n\n{text}\n",encoding="utf-8")
    return path


def list_report_archive() -> List[Path]:
    try: return sorted(REPORT_DIR.glob("*.md"),key=lambda p:p.stat().st_mtime,reverse=True)
    except Exception: return []

# ============================================================
# Charts
# ============================================================
def price_chart(df:pd.DataFrame,ticker:str)->go.Figure:
    d=add_indicators(df); fig=go.Figure()
    fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name=ticker,increasing_line_color="#12a878",decreasing_line_color="#e05260"))
    for col in ["MA9","MA21","MA50","MA200"]:
        if col in d: fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col,line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=d.index,y=d.BBUpper,name="BB Upper",line=dict(width=1,dash="dot",color="#8cbcf0")))
    fig.add_trace(go.Scatter(x=d.index,y=d.BBLower,name="BB Lower",line=dict(width=1,dash="dot",color="#8cbcf0")))
    fig.update_layout(template="plotly_white",height=470,margin=dict(l=0,r=0,t=18,b=0),xaxis_rangeslider_visible=False,legend=dict(orientation="h",y=1.02),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
    return fig


def target_chart(current:float,low:float,avg:float,high:float)->go.Figure:
    fig=go.Figure()
    vals=[v for v in [low,avg,high] if np.isfinite(v)]
    labels=["Low","Average","High"][:len(vals)]
    if vals: fig.add_trace(go.Bar(x=labels,y=vals,text=[f"${v:.2f}" for v in vals],textposition="auto",marker_color=["#e8a0ad","#8bc5f2","#3aa7ff"]))
    if np.isfinite(current): fig.add_hline(y=current,line_dash="dash",line_color="#173f69",annotation_text=f"Current ${current:.2f}")
    fig.update_layout(template="plotly_white",height=300,margin=dict(l=0,r=0,t=15,b=0),showlegend=False,paper_bgcolor="rgba(0,0,0,0)")
    return fig


def decision_forecast_chart(score_value: float, label: str, current: float, low: float, avg: float, high: float) -> go.Figure:
    """Professional decision gauge + 12-month analyst target range."""
    fig = go.Figure()
    score_value = safe_float(score_value)
    current = safe_float(current); low = safe_float(low); avg = safe_float(avg); high = safe_float(high)
    if np.isfinite(score_value):
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=score_value,
            title={"text": f"<b>{label or 'Signal'}</b><br><span style='font-size:10px'>Quantitative decision score</span>"},
            number={"suffix":"/100", "font":{"size":28,"color":"#173f69"}},
            gauge={"axis":{"range":[0,100],"tickfont":{"size":9}},
                   "bar":{"color":"#1687f8","thickness":0.28},
                   "bgcolor":"#f5faff","borderwidth":0,
                   "steps":[{"range":[0,45],"color":"#ffecef"},{"range":[45,70],"color":"#fff6df"},{"range":[70,100],"color":"#e8faf3"}],
                   "threshold":{"line":{"color":"#173f69","width":3},"thickness":0.8,"value":score_value}}
        ))
    fig.update_layout(height=255, margin=dict(l=10,r=10,t=35,b=5), paper_bgcolor="rgba(0,0,0,0)", font={"color":"#173f69"})
    return fig


def target_range_chart(current: float, low: float, avg: float, high: float) -> go.Figure:
    current=safe_float(current); low=safe_float(low); avg=safe_float(avg); high=safe_float(high)
    fig=go.Figure()
    vals=[v for v in [low,high,current,avg] if np.isfinite(v)]
    if vals:
        mn=min(vals); mx=max(vals); pad=max((mx-mn)*0.18, max(abs(mx),1)*0.03); xmin=mn-pad; xmax=mx+pad
        if np.isfinite(low) and np.isfinite(high):
            fig.add_trace(go.Scatter(x=[low,high],y=[0,0],mode="lines",line=dict(color="#8fc7f5",width=14),name="Analyst range",hoverinfo="skip"))
        if np.isfinite(avg):
            fig.add_trace(go.Scatter(x=[avg],y=[0],mode="markers+text",text=[f"Avg ${avg:,.2f}"],textposition="top center",marker=dict(size=15,color="#1687f8",line=dict(width=3,color="#fff")),name="Average target"))
        if np.isfinite(current):
            fig.add_trace(go.Scatter(x=[current],y=[0],mode="markers+text",text=[f"Current ${current:,.2f}"],textposition="bottom center",marker=dict(size=13,color="#173f69",symbol="diamond"),name="Current"))
        if np.isfinite(low): fig.add_annotation(x=low,y=0,text=f"Low ${low:,.2f}",showarrow=False,yshift=25,font=dict(size=10,color="#7d91a6"))
        if np.isfinite(high): fig.add_annotation(x=high,y=0,text=f"High ${high:,.2f}",showarrow=False,yshift=25,font=dict(size=10,color="#0a8f6a"))
        fig.update_xaxes(range=[xmin,xmax],showgrid=False,zeroline=False,tickprefix="$",tickformat=",.0f")
    fig.update_yaxes(visible=False,range=[-1,1]); fig.update_layout(height=145,margin=dict(l=8,r=8,t=12,b=15),showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig


def score_gauge(score:int)->go.Figure:
    fig=go.Figure(go.Indicator(mode="gauge+number",value=score,number={"font":{"size":30,"color":"#173f69"}},gauge={"axis":{"range":[0,100]},"bar":{"color":"#1687f8"},"bgcolor":"#eef7ff","borderwidth":0,"steps":[{"range":[0,45],"color":"#fff0f2"},{"range":[45,70],"color":"#fff7e7"},{"range":[70,100],"color":"#e8faf3"}]}))
    fig.update_layout(height=190,margin=dict(l=15,r=15,t=10,b=5),paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("<div class='side-brand'><div class='logo-mark'><span></span><span></span><span></span><b>AI</b></div><div><div class='side-brand-title'>AI Stock Analyzer</div><div class='side-brand-sub'>Live market research • Quant signals • AI insights</div></div></div>",unsafe_allow_html=True)

sidebar_ticker=st.sidebar.text_input("Search stock / ticker",value=st.session_state.selected_symbol,key="sidebar_stock_search",placeholder="AAPL, NVDA, MSFT...").upper().strip()
if sidebar_ticker and sidebar_ticker != st.session_state.selected_symbol:
    st.session_state.selected_symbol=sidebar_ticker
    log_stock_event(sidebar_ticker,"Search")
symbol=st.session_state.selected_symbol or "NVDA"

market_filter=st.sidebar.selectbox("Market cap",["Any","$2B+","$10B+","$50B+"],index=1)
sector_filter=st.sidebar.selectbox("Sector",["All Sectors","Technology","Communication Services","Consumer Cyclical","Financial Services","Healthcare","Energy","Industrials"])
period=st.sidebar.selectbox("Price history",["3mo","6mo","1y","2y","5y"],index=2)
chart_interval=st.sidebar.selectbox("Chart interval",["5m","15m","30m","1h","1d","1wk","1mo"],index=4)
news_limit=st.sidebar.slider("News stories",5,20,10)
show_watchlist=st.sidebar.checkbox("Show watchlist only",value=False)

st.sidebar.markdown("### Earnings")
earnings_window=st.sidebar.selectbox("Calendar window",["Next 7 days","Next 14 days","Next 30 days"],index=1)

st.sidebar.markdown("### Auto refresh")
auto_refresh=st.sidebar.checkbox("Enable auto refresh",value=True)
refresh_seconds=st.sidebar.selectbox("Refresh interval",[60,120,300,600],index=2,format_func=lambda x:f"Every {x//60} minute" if x>=60 else f"Every {x}s")
if auto_refresh and st_autorefresh:
    st_autorefresh(interval=refresh_seconds*1000,key="v8_auto_refresh")

# ============================================================
# Global data
# ============================================================
info=get_info(symbol); df=get_history(symbol,period,chart_interval); news=combined_news(symbol,news_limit); regime=market_regime(); score=score_stock(df,info,news)

# ============================================================
# Header
# ============================================================
st.markdown(
    """<div class='header'><div class='brand-wrap'><div class='logo-mark logo-large'><span></span><span></span><span></span><b>AI</b></div><div class='brand'>AI Stock Analyzer<small>Market research, quantitative signals &amp; AI-assisted insights</small></div></div></div>""",
    unsafe_allow_html=True,
)

# Market strip — each instrument is a single visual field and the ticker itself is clickable
market_data=[]
for t in MARKET_TICKERS:
    h=get_history(t,"1mo","1d")
    if h.empty:
        market_data.append((t,np.nan,np.nan,pd.Series(dtype=float)))
        continue
    close=pd.to_numeric(h["Close"],errors="coerce").dropna(); p=safe_float(close.iloc[-1]); prev=safe_float(close.iloc[-2]) if len(close)>1 else p; ch=((p/prev)-1)*100 if prev else np.nan
    market_data.append((t,p,ch,close.tail(20)))
market_cols=st.columns(8,gap="small")
for i,(t,p,ch,series) in enumerate(market_data):
    with market_cols[i]:
        st.markdown("<div class='market-cell'>",unsafe_allow_html=True)
        st.button(t, key=f"market_pick_{t}", use_container_width=True, help=f"Open {t} details", on_click=open_stock_details, args=(t,"Market instrument"))
        cls="hero-up" if np.isfinite(ch) and ch>=0 else "hero-down"
        price_txt=f"{p:,.2f}" if np.isfinite(p) else "—"
        ch_txt=f"{ch:+.2f}%" if np.isfinite(ch) else "—"
        st.markdown(f"<div class='market-card-price'>{price_txt}</div><div class='market-card-change {cls}'>{ch_txt}</div>",unsafe_allow_html=True)
        if len(series)>1:
            sf=go.Figure(go.Scatter(x=list(range(len(series))),y=series.values,mode="lines",line=dict(width=1.6,color="#1687f8")))
            sf.update_layout(height=52,margin=dict(l=0,r=0,t=2,b=0),xaxis=dict(visible=False),yaxis=dict(visible=False),showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(sf,use_container_width=True,key=f"market_spark_{t}",config={"displayModeBar":False})
        st.markdown("</div>",unsafe_allow_html=True)

# Stock detail dialog -------------------------------------------------
@st.dialog("Stock Details", width="large")
def stock_detail_dialog(ticker: str):
    t = str(ticker).upper().strip()
    di = get_info(t)
    dh = get_history(t, "6mo", "1d")
    dn = get_news(t, 8)
    ds = score_stock(dh, di, dn)
    current = safe_float(di.get("currentPrice", ds.get("price")))
    low = safe_float(di.get("targetLowPrice")); avg = safe_float(di.get("targetMeanPrice")); high = safe_float(di.get("targetHighPrice"))
    company = di.get("shortName", t)
    st.markdown(f"### {t}  ·  {company}")
    st.caption("Stock details • quantitative signals • analyst targets • risk context")
    r1,r2,r3,r4,r5 = st.columns(5)
    r1.metric("Price", fmt_money(current))
    r2.metric("1D", fmt_pct(ds.get("momentum")))
    r3.metric("Signal", str(ds.get("label","—")))
    r4.metric("Score", f"{int(safe_float(ds.get('score',0))):d}/100")
    r5.metric("Beta", fmt_num(di.get("beta")))
    st.markdown("#### Price & decision")
    if not dh.empty:
        st.plotly_chart(price_chart(dh, t), use_container_width=True, key=f"dialog_price_{t}", config={"displayModeBar":False})
    if np.isfinite(current) and any(np.isfinite(x) for x in [low,avg,high]):
        g1,g2=st.columns([1,1.6],gap="medium")
        with g1:
            decision = "BUY" if safe_float(ds.get("score")) >= 72 else "HOLD" if safe_float(ds.get("score")) >= 45 else "SELL"
            st.plotly_chart(decision_forecast_chart(ds.get("score",50),decision,current,low,avg,high),use_container_width=True,key=f"dialog_decision_{t}",config={"displayModeBar":False})
        with g2:
            st.plotly_chart(target_range_chart(current,low,avg,high),use_container_width=True,key=f"dialog_target_{t}",config={"displayModeBar":False})
    st.markdown("#### Key fundamentals & risk")
    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Market Cap", fmt_money(di.get("marketCap")))
    f2.metric("P/E", fmt_num(di.get("trailingPE")))
    f3.metric("RSI", fmt_num(ds.get("rsi")))
    f4.metric("20D Momentum", fmt_pct(ds.get("momentum")))
    st.markdown("#### Analyst targets")
    a1,a2,a3,a4=st.columns(4)
    a1.metric("Current",fmt_money(current)); a2.metric("Target Low",fmt_money(low)); a3.metric("Target Avg",fmt_money(avg)); a4.metric("Target High",fmt_money(high))
    st.markdown("#### Recent news")
    if dn:
        for n in dn[:5]:
            st.markdown(f"**{n.get('title','')}**")
            st.caption(f"{n.get('publisher','News')} • {n.get('published','')}")
    else:
        st.caption("No recent news available.")
    if st.button("Close", key=f"dialog_close_{t}", type="secondary", use_container_width=True):
        st.session_state.stock_detail_open = False
        st.rerun()

# Tabs
T=st.tabs(["🧠 Daily Intelligence","🏠 Dashboard","🤖 AI Scanner","📰 Stories","🔥 Movers","⚖️ Compare","🔎 Stock Analysis","⭐ Analyst","📅 Earnings","📊 Technical + Risk","🎯 Options","🚀 IPOs","🎓 Stock Education","🧾 Activity Log"])

# ============================================================
# Reusable news rail
# ============================================================
def render_news_rail(title="Top Stories",ticker=""):
    if st.session_state.hide_right_rail:
        return
    st.markdown(f"<div class='card'><div class='card-title'>📰 {title}</div><div class='card-sub'>Live-style market stories • sentiment-tagged</div></div>",unsafe_allow_html=True)
    items=combined_news(ticker,10) if ticker else market_news()
    if not items:
        st.info("News feed temporarily unavailable.")
        return
    for idx,n in enumerate(items[:10]):
        label,_=classify_sentiment(n.get("title","")); cls="sent-pos" if label=="Positive" else "sent-neg" if label=="Negative" else "sent-neu"
        nt=str(n.get("ticker","")).upper().strip()
        nc1,nc2=st.columns([.22,.78],gap="small")
        with nc1:
            if nt:
                clickable_ticker(nt,f"rail_{title.replace(chr(32), chr(95))}_{idx}_{nt}","Open",action="News ticker")
        with nc2:
            st.markdown(f"<div class='news-card'><span class='news-sent {cls}'>{label}</span><div class='news-title'>{n.get('title','')}</div><div class='news-meta'>{n.get('publisher','News')} • {nt} • {n.get('published','')}</div></div>",unsafe_allow_html=True)
            if n.get("url"):
                st.markdown(f"<div style='margin:-3px 0 7px 2px;font-size:10px'><a href='{n['url']}' target='_blank'>Read story →</a></div>",unsafe_allow_html=True)


def render_hero():
    price=score.get("price",safe_float(info.get("currentPrice"))); prev=safe_float(df["Close"].iloc[-2]) if len(df)>1 else price; change=((price/prev)-1)*100 if np.isfinite(price) and np.isfinite(prev) and prev else np.nan
    company=info.get("shortName",symbol); sector=info.get("sector","—")
    cls="hero-up" if np.isfinite(change) and change>=0 else "hero-down"
    st.markdown(f"""<div class='hero-card'><div style='display:flex;justify-content:space-between;gap:20px;align-items:center'><div><div class='hero-ticker'>{symbol} <span class='badge'>{sector}</span><span class='badge'>Market Cap {fmt_money(info.get('marketCap'))}</span></div><div class='hero-company'>{company}</div><div class='hero-price'>{'$'+f'{price:,.2f}' if np.isfinite(price) else '—'} <span class='{cls}'>{f'{change:+.2f}%' if np.isfinite(change) else ''}</span></div><div class='card-sub'>Updated {datetime.now().strftime('%b %d, %Y %I:%M %p')}</div></div><div style='text-align:right'><div class='score'> {score.get('score',0)}<span style='font-size:13px;color:#7d91a6'>/100</span></div><div class='score-label'>{score.get('label','—')}</div><span class='badge-green badge'>Market {regime['label']}</span></div></div></div>""",unsafe_allow_html=True)

# ============================================================
# Dashboard
# ============================================================
def page_columns():
    if st.session_state.hide_right_rail:
        return st.container(), st.container()
    if st.session_state.focus_mode:
        return st.columns([3.0,0.8],gap="medium")
    return st.columns([2.2,.9],gap="medium")

with T[1]:
    main,right=st.columns([3.3,0.7],gap="medium") if st.session_state.focus_mode else st.columns([2.25,0.92],gap="medium")
    with main:
        render_hero()
        k=st.columns(6)
        k[0].markdown(f"<div class='kpi'><div class='kpi-label'>Market Cap</div><div class='kpi-value'>{fmt_money(info.get('marketCap'))}</div></div>",unsafe_allow_html=True)
        k[1].markdown(f"<div class='kpi'><div class='kpi-label'>P/E</div><div class='kpi-value'>{fmt_num(info.get('trailingPE'))}</div></div>",unsafe_allow_html=True)
        k[2].markdown(f"<div class='kpi'><div class='kpi-label'>Forward P/E</div><div class='kpi-value'>{fmt_num(info.get('forwardPE'))}</div></div>",unsafe_allow_html=True)
        k[3].markdown(f"<div class='kpi'><div class='kpi-label'>RSI (14)</div><div class='kpi-value'>{fmt_num(score.get('rsi'))}</div></div>",unsafe_allow_html=True)
        k[4].markdown(f"<div class='kpi'><div class='kpi-label'>Rel Volume</div><div class='kpi-value'>{fmt_num(score.get('relvol'))}x</div></div>",unsafe_allow_html=True)
        k[5].markdown(f"<div class='kpi'><div class='kpi-label'>20D Momentum</div><div class='kpi-value'>{fmt_pct(score.get('momentum'))}</div></div>",unsafe_allow_html=True)
        st.markdown("<div class='section-rule'></div>",unsafe_allow_html=True)
        chart_col,gauge_col=st.columns([2.35,.65])
        with chart_col:
            st.markdown(f"<div class='card-title'>📈 {symbol} Price & Momentum</div>",unsafe_allow_html=True)
            if df.empty: st.warning("No price history available.")
            else: st.plotly_chart(price_chart(df,symbol),use_container_width=True,key="v5_dashboard_price")
        with gauge_col:
            st.markdown("<div class='card-title'>AI Composite</div>",unsafe_allow_html=True)
            st.plotly_chart(score_gauge(score.get("score",0)),use_container_width=True,key="v5_score_gauge")
            st.markdown(f"<div class='hint'><b>Signal:</b> {score.get('label')}<br>News sentiment: {score.get('news_score',0):+d}<br>Market regime: {regime['label']}</div>",unsafe_allow_html=True)
        g1,g2,g3=st.columns(3)
        with g1:
            st.markdown("<div class='card'><div class='card-title'>🟢 Top Gainers</div>",unsafe_allow_html=True)
            g=safe_market_cap_filter(get_screener("day_gainers",8),market_filter)
            if g.empty: st.caption("Gainer feed unavailable")
            else:
                for i,row in g.head(5).iterrows():
                    t=str(row.get("symbol","")); ch=safe_float(row.get("regularMarketChangePercent")); clickable_ticker(t,f"gain_{i}",fmt_pct(ch))
            st.markdown("</div>",unsafe_allow_html=True)
        with g2:
            st.markdown("<div class='card'><div class='card-title'>🔴 Top Losers</div>",unsafe_allow_html=True)
            l=safe_market_cap_filter(get_screener("day_losers",8),market_filter)
            if l.empty: st.caption("Loser feed unavailable")
            else:
                for i,row in l.head(5).iterrows():
                    t=str(row.get("symbol","")); ch=safe_float(row.get("regularMarketChangePercent")); clickable_ticker(t,f"lose_{i}",fmt_pct(ch))
            st.markdown("</div>",unsafe_allow_html=True)
        with g3:
            st.markdown("<div class='card'><div class='card-title'>📅 Earnings</div>",unsafe_allow_html=True)
            ed=normalize_earnings(earnings_events(7))
            if ed.empty: st.caption("Earnings feed unavailable")
            else:
                for i,row in ed.head(6).iterrows():
                    t=str(row.get("Ticker",row.get("symbol",""))); dt=str(row.get("Earnings Date",""))[:10]; clickable_ticker(t,f"dash_earn_{i}",dt)
            st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='card-title'>⚡ Market Regime & Risk</div>",unsafe_allow_html=True)
        r1,r2,r3,r4=st.columns(4); r1.metric("Regime",regime["label"]); r2.metric("Regime score",f"{regime['score']}/100"); r3.metric("Beta",fmt_num(info.get("beta"))); r4.metric("20D momentum",fmt_pct(score.get("momentum")))
        st.dataframe(regime["table"],use_container_width=True,hide_index=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with right:
        render_news_rail("Latest Markets News")
        st.markdown("<div class='card'><div class='card-title'>🎯 Quick Insight</div><div class='hint'>Click any ticker in the dashboard to make it the active stock. All analysis tabs will follow the selected ticker.</div></div>",unsafe_allow_html=True)

# ============================================================
# AI Daily Intelligence Center
# ============================================================
with T[0]:
    st.markdown("<div class='intelligence-hero'><div><div class='hero-ticker'>🧠 AI Daily Intelligence Center</div><div class='hero-company'>Morning briefing • ranked opportunities • catalysts • risk • historical signal tracking</div></div><div class='ai-live'>● LIVE RESEARCH</div></div>",unsafe_allow_html=True)
    q1,q2,q3,q4,q5=st.columns(5)
    with q1:
        daily_provider=st.selectbox("AI Analyst",["Auto","OpenAI","Claude","Local Ollama","Disabled"],index=0,key="daily_provider")
    with q2: daily_model=st.text_input("Model",value=OPENAI_MODEL_DEFAULT if daily_provider in ["Auto","OpenAI"] else CLAUDE_MODEL_DEFAULT if daily_provider=="Claude" else "llama3.2:3b",key="daily_model")
    with q3: daily_universe=st.selectbox("Universe",[100,250,350],index=1,key="daily_universe")
    with q4: daily_top=st.selectbox("Top list",[5,10],index=1,key="daily_top")
    with q5: daily_enrich=st.selectbox("Deep enrich",[20,30,40,60],index=2,key="daily_enrich")
    a1,a2,a3=st.columns([1.2,1.2,2.2])
    with a1: run_daily=st.button("🚀 Run Today's Intelligence",type="primary",use_container_width=True,key="run_daily")
    with a2: auto_daily=st.toggle("Auto-run if today's report is missing",value=False,key="auto_daily")
    with a3: st.markdown("<div class='hint'>For unattended scheduling, use the included <b>scripts/daily_scan.py</b> + GitHub Actions workflow or Windows Task Scheduler. The web app itself should not be relied on as a background scheduler.</div>",unsafe_allow_html=True)

    if run_daily or (auto_daily and not st.session_state.daily_scan_done):
        with st.spinner("Scanning the market and building today's intelligence report…"):
            u=tuple(SCANNER_UNIVERSE[:daily_universe])
            scan=run_ai_market_scan(u,daily_enrich,regime["label"],regime["score"])
            if not scan.empty:
                scan=scan.groupby("Decision",group_keys=False).head(daily_top).copy()
                _write_recommendation_history(scan)
            sectors=sector_rotation_snapshot()
            actions=analyst_upgrade_downgrade_snapshot(scan["Ticker"].tolist() if not scan.empty else WATCHLIST)
            catalysts=news_catalyst_snapshot(scan["Ticker"].tolist() if not scan.empty else WATCHLIST)
            st.session_state.daily_scan_df=scan
            st.session_state.daily_ai_report=generate_daily_intelligence(scan,regime,sectors,actions,catalysts,daily_provider,daily_model)
            st.session_state.daily_scan_done=True
            save_intelligence_report(st.session_state.daily_ai_report)

    scan=st.session_state.daily_scan_df
    if scan.empty:
        st.markdown("<div class='card'><div class='card-title'>☀️ Morning Market Briefing</div><div class='card-sub'>Run today's intelligence to populate the AI briefing, ranked lists, catalysts and historical tracking.</div></div>",unsafe_allow_html=True)
        st.info("No daily scan has been run in this session yet. The AI Scanner tab can also be used independently.")
    else:
        # Headline metrics
        buys=int((scan.Decision=="BUY").sum()); watches=int((scan.Decision=="WATCH").sum()); avoids=int((scan.Decision=="AVOID").sum())
        m=st.columns(5); m[0].metric("Market Regime",regime["label"]); m[1].metric("BUY",buys); m[2].metric("WATCH",watches); m[3].metric("AVOID",avoids); m[4].metric("Universe scanned",f"{daily_universe}")
        if st.session_state.daily_ai_report:
            st.markdown("### ☀️ Morning Market Briefing")
            st.markdown(st.session_state.daily_ai_report)

        sectors=sector_rotation_snapshot()
        s1,s2=st.columns(2)
        with s1:
            st.markdown("### 🔄 Sector Rotation")
            if sectors.empty: st.caption("Sector data unavailable")
            else: st.dataframe(sectors.head(11),use_container_width=True,hide_index=True)
        with s2:
            st.markdown("### 🧨 Unusual Volume")
            uv=scan[pd.to_numeric(scan["RelVol"],errors="coerce")>=1.5].sort_values("RelVol",ascending=False)
            if uv.empty: st.caption("No unusual-volume candidates in today's enriched set.")
            else:
                for i,r in uv.head(8).iterrows(): clickable_ticker(str(r.Ticker),f"daily_uv_{i}",f"{safe_float(r.RelVol):.1f}x")

        o1,o2=st.columns(2)
        with o1:
            st.markdown("### 🎯 Unusual Options")
            opt=unusual_options_snapshot(scan["Ticker"].tolist(),12)
            if opt.empty:
                st.caption("Options activity unavailable or no unusually active contracts detected.")
            else:
                for oi,ot in enumerate(opt.get("Ticker",pd.Series(dtype=str)).dropna().astype(str).unique()):
                    clickable_ticker(ot,f"daily_opt_{oi}","Open options",action="Unusual options")
                st.dataframe(opt.drop(columns=["Ticker"],errors="ignore"),use_container_width=True,hide_index=True)
        with o2:
            st.markdown("### 📅 Earnings Catalysts")
            earn=normalize_earnings(earnings_events(14))
            if earn.empty:
                st.caption("Earnings feed unavailable.")
            else:
                for ei,er in earn.head(15).iterrows():
                    et=str(er.get("Ticker","")); edt=str(er.get("Earnings Date",""))[:10]
                    clickable_ticker(et,f"daily_earn_{ei}",edt,action="Earnings")
                    extras=[]
                    for ec in ["EPS Est","EPS Actual","Time"]:
                        if ec in er and pd.notna(er[ec]): extras.append(f"{ec}: {er[ec]}")
                    if extras: st.caption(" • ".join(extras))

        n1,n2=st.columns(2)
        with n1:
            st.markdown("### 📰 News Catalysts")
            cat=news_catalyst_snapshot(scan["Ticker"].tolist(),10)
            if cat.empty:
                st.caption("No news catalysts returned.")
            else:
                for ni,nt in enumerate(cat.get("Ticker",pd.Series(dtype=str)).dropna().astype(str).unique()):
                    clickable_ticker(nt,f"daily_cat_{ni}","Open stock",action="News catalyst")
                st.dataframe(cat.drop(columns=["Ticker"],errors="ignore"),use_container_width=True,hide_index=True)
        with n2:
            st.markdown("### ⬆️ Biggest Upgrades / Downgrades")
            act=analyst_upgrade_downgrade_snapshot(scan["Ticker"].tolist())
            if act.empty:
                st.caption("No detailed analyst actions returned. Yahoo consensus remains available in the Analyst tab.")
            else:
                for ai,at in enumerate(act.get("Ticker",pd.Series(dtype=str)).dropna().astype(str).unique()):
                    clickable_ticker(at,f"daily_act_{ai}","Open analyst view",action="Analyst action")
                st.dataframe(act.drop(columns=["Ticker"],errors="ignore"),use_container_width=True,hide_index=True)

        st.markdown("### 🏆 AI Top 10")
        for decision,title,icon in [("BUY","Top BUY","🟢"),("WATCH","Top WATCH","🟡"),("AVOID","Top AVOID","🔴")]:
            st.markdown(f"#### {icon} {title}")
            part=scan[scan.Decision==decision].sort_values("Score",ascending=False).head(daily_top)
            if part.empty: st.caption("None")
            else:
                cc=st.columns(2)
                for j,(_,r) in enumerate(part.iterrows()):
                    with cc[j%2]:
                        st.markdown(f"<div class='rank-card'><div class='rank-top'><b>{r.Ticker}</b><span class='rank-score'>{int(r.Score)}/100</span></div><div class='rank-reason'>{r.Reason}</div><div class='rank-meta'>Tech {int(r.Technical)} • Fund {int(r.Fundamental)} • Earn {int(r.Earnings)} • News {int(r.News)} • Vol {int(r.Volume)} • Opt {int(r.Options)}</div></div>",unsafe_allow_html=True)
                        st.button(f"Open {r.Ticker}",key=f"daily_open_{decision}_{j}",use_container_width=True,on_click=open_stock_details,args=(r.Ticker,"Daily Intelligence"))

        perf=historical_recommendation_performance()
        st.markdown("### 📈 Historical Performance of Previous AI Recommendations")
        if perf.empty: st.caption("Performance history will appear after daily recommendations have been saved and enough market history is available.")
        else:
            pcols=st.columns(4)
            valid=perf[pd.to_numeric(perf["Return %"],errors="coerce").notna()]
            pcols[0].metric("Tracked signals",len(valid))
            pcols[1].metric("Avg current return",fmt_pct(valid["Return %"].mean()) if not valid.empty else "—")
            pcols[2].metric("Positive signals",f"{(valid['Return %']>0).mean()*100:.0f}%" if not valid.empty else "—")
            pcols[3].metric("BUY avg",fmt_pct(valid.loc[valid.Decision=="BUY","Return %"].mean()) if not valid.loc[valid.Decision=="BUY"].empty else "—")
            st.dataframe(perf.sort_values("Date",ascending=False).head(50),use_container_width=True,hide_index=True)

        st.markdown("### 🗂️ Daily Report Archive")
        archive=list_report_archive()
        if archive:
            for pth in archive[:15]:
                st.markdown(f"📄 **{pth.name}** — {datetime.fromtimestamp(pth.stat().st_mtime).strftime('%Y-%m-%d %H:%M')} ")
                with st.expander("Preview",expanded=False): st.markdown(pth.read_text(encoding="utf-8")[:12000])
        else: st.caption("No archived reports yet.")

# ============================================================
# AI Market Scanner
# ============================================================
with T[2]:
    st.markdown("<div class='scanner-banner'><div class='card-title'>🤖 AI Market Scanner</div><div class='card-sub'>Quantitative candidate discovery first; AI Daily Intelligence then explains the highest-conviction opportunities.</div></div>",unsafe_allow_html=True)
    sc1,sc2,sc3,sc4,sc5=st.columns([1.1,1.1,1.1,1.1,1.5])
    with sc1: universe_size=st.selectbox("Universe",[100,250,350],index=1,key="scanner_universe")
    with sc2: top_n=st.selectbox("Show Top",[5,10],index=1,key="scanner_top")
    with sc3: scanner_enrich=st.selectbox("Deep enrich",[20,30,40,60],index=2,key="scanner_enrich")
    with sc4: scanner_market=st.selectbox("Market cap",["Any","$2B+","$10B+"],index=1,key="scanner_market")
    with sc5: st.markdown("<div class='scanner-mini'>Weights</div><div style='font-size:10px;color:#55728f'>Tech 25% • Fund 20% • Earnings 15% • News 15% • Volume 10% • Options 5% • Regime 10%</div>",unsafe_allow_html=True)
    run_scan=st.button("🔄 Run AI Market Scan Now",type="primary",key="run_scanner")
    if run_scan:
        with st.spinner("Scanning hundreds of symbols…"):
            scanner_rows=run_ai_market_scan(tuple(SCANNER_UNIVERSE[:universe_size]),scanner_enrich,regime["label"],regime["score"])
            if scanner_market!="Any" and not scanner_rows.empty:
                threshold={"$2B+":2e9,"$10B+":10e9}[scanner_market]
                keep=[]
                for t in scanner_rows.Ticker:
                    inf=get_info(t); cap=safe_float(inf.get("marketCap"))
                    if np.isfinite(cap) and cap>=threshold: keep.append(t)
                scanner_rows=scanner_rows[scanner_rows.Ticker.isin(keep)].copy()
            _write_recommendation_history(scanner_rows.head(30))
            st.session_state.daily_scan_df=scanner_rows.copy()
            st.session_state.daily_scan_done=True
            st.rerun()
    scanner_rows=st.session_state.daily_scan_df if not st.session_state.daily_scan_df.empty else run_ai_market_scan(tuple(SCANNER_UNIVERSE[:universe_size]),scanner_enrich,regime["label"],regime["score"])
    if scanner_rows.empty:
        st.warning("No scanner data returned. Yahoo/yfinance may be rate-limiting; try again later or reduce Deep enrich.")
    else:
        for decision,title,icon in [("BUY","Top BUY Opportunities","🟢"),("WATCH","Top WATCH Opportunities","🟡"),("AVOID","Top AVOID Opportunities","🔴")]:
            st.markdown(f"### {icon} {title}")
            part=scanner_rows[scanner_rows.Decision==decision].sort_values("Score",ascending=False).head(top_n)
            if part.empty: st.caption("No candidates in this category.")
            else:
                for i,(_,r) in enumerate(part.iterrows(),1):
                    c1,c2,c3,c4,c5=st.columns([.9,1.0,1.0,1.0,2.4])
                    with c1:
                        st.button(str(r.Ticker),key=f"scan8_{decision}_{i}",on_click=open_stock_details,args=(r.Ticker,"AI Scanner"))
                    with c2: st.metric("Score",f"{int(r.Score)}/100")
                    with c3: st.metric("Momentum",fmt_pct(r.Momentum))
                    with c4: st.metric("RelVol",f"{safe_float(r.RelVol):.2f}x")
                    with c5: st.markdown(f"**{r.Reason}**  ")
                    st.caption(f"Tech {int(r.Technical)} • Fund {int(r.Fundamental)} • Earnings {int(r.Earnings)} • News {int(r.News)} • Volume {int(r.Volume)} • Options {int(r.Options)} • {r['Market Regime']}")
# ============================================================
# Stories
# ============================================================
with T[3]:
    left,right=page_columns()
    with left:
        st.markdown("<div class='card-title'>📰 Top Stories / Latest Markets News</div>",unsafe_allow_html=True)
        for i,n in enumerate(market_news()[:25]):
            label,_=classify_sentiment(n.get("title","")); t=n.get("ticker","")
            c1,c2=st.columns([.12,.88]);
            with c1:
                if t: clickable_ticker(t,f"story_{i}",t)
            with c2:
                st.markdown(f"**{n.get('title','')}**")
                st.caption(f"{n.get('publisher','News')} • {label} • {n.get('published','')}")
                if n.get("url"): st.markdown(f"[Read story →]({n['url']})")
            st.divider()
    with right: render_news_rail("Story Feed",symbol)

# ============================================================
# Movers
# ============================================================
with T[4]:
    left,right=page_columns()
    with left:
        st.markdown("<div class='card-title'>🔥 Trending Tickers — Top Gainers / Top Losers</div>",unsafe_allow_html=True)
        mg=safe_market_cap_filter(get_screener("day_gainers",20),market_filter); ml=safe_market_cap_filter(get_screener("day_losers",20),market_filter)
        a,b=st.columns(2)
        for container,dfm,title,prefix in [(a,mg,"🟢 Top Gainers","vg"),(b,ml,"🔴 Top Losers","vl")]:
            with container:
                st.markdown(f"<div class='card'><div class='card-title'>{title}</div>",unsafe_allow_html=True)
                if dfm.empty: st.warning("Mover data unavailable. Try refresh or another market-cap filter.")
                else:
                    for i,row in dfm.head(12).iterrows():
                        t=str(row.get("symbol","")); ch=safe_float(row.get("regularMarketChangePercent")); p=safe_float(row.get("regularMarketPrice")); vol=safe_float(row.get("regularMarketVolume"));
                        clickable_ticker(t,f"{prefix}_{i}",f"${p:.2f}  {ch:+.2f}%  Vol {vol/1e6:.1f}M" if np.isfinite(p) and np.isfinite(ch) else "")
                        reason,headline=mover_reason(t)
                        st.caption(f"{reason}: {headline}")
                st.markdown("</div>",unsafe_allow_html=True)
    with right: render_news_rail("Mover Catalysts")

# ============================================================
# Compare
# ============================================================
with T[5]:
    left,right=page_columns()
    with left:
        st.markdown("<div class='card-title'>⚖️ Stock Comparison</div>",unsafe_allow_html=True)
        raw=st.text_input("Compare up to 8 tickers",value=st.session_state.compare_tickers)
        st.session_state.compare_tickers=raw
        tickers=[x.strip().upper() for x in raw.split(",") if x.strip()][:8]
        rows=[]
        for t in tickers:
            d=get_history(t,"1y","1d"); inf=get_info(t); ns=get_news(t,6); sc=score_stock(d,inf,ns)
            rows.append({"Ticker":t,"Score":sc.get("score"),"Signal":sc.get("label"),"Price":sc.get("price"),"1D %":safe_float(add_indicators(d).iloc[-1]["Ret1D"]) if not d.empty else np.nan,"RSI":sc.get("rsi"),"RelVol":sc.get("relvol"),"Market Cap":inf.get("marketCap"),"P/E":inf.get("trailingPE"),"Rev Growth":safe_float(inf.get("revenueGrowth"))*100})
        comp=pd.DataFrame(rows)
        if not comp.empty:
            st.markdown("<div class='card'>",unsafe_allow_html=True)
            cols=st.columns(min(len(tickers),8))
            for i,t in enumerate(tickers): clickable_ticker(t,f"cmp_pick_{i}","Open insight")
            st.dataframe(comp,use_container_width=True,hide_index=True)
            fig=go.Figure(go.Bar(x=comp["Ticker"],y=comp["Score"],text=comp["Score"],textposition="auto",marker_color="#2d9bf0")); fig.update_layout(template="plotly_white",height=330,margin=dict(l=0,r=0,t=20,b=0),yaxis_range=[0,100],paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True,key="v5_compare_score")
            st.markdown("</div>",unsafe_allow_html=True)
    with right: render_news_rail("Comparison News",symbol)

# ============================================================
# Stock analysis
# ============================================================
with T[6]:
    left,right=page_columns()
    with left:
        render_hero()
        st.markdown("<div class='card'><div class='card-title'>🔎 Complete Stock Insight</div>",unsafe_allow_html=True)
        a,b,c,d4=st.columns(4); a.metric("Recommendation",score.get("label")); b.metric("Sector",info.get("sector","—")); c.metric("Industry",info.get("industry","—")); d4.metric("Beta",fmt_num(info.get("beta")))
        st.markdown("#### Fundamentals")
        fund=pd.DataFrame([
            ["Market Cap",fmt_money(info.get("marketCap"))],["Trailing P/E",fmt_num(info.get("trailingPE"))],["Forward P/E",fmt_num(info.get("forwardPE"))],["PEG",fmt_num(info.get("pegRatio"))],
            ["Revenue Growth",fmt_pct(safe_float(info.get("revenueGrowth"))*100)],["Profit Margin",fmt_pct(safe_float(info.get("profitMargins"))*100)],["ROE",fmt_pct(safe_float(info.get("returnOnEquity"))*100)],
            ["52W High",fmt_money(info.get("fiftyTwoWeekHigh"))],["52W Low",fmt_money(info.get("fiftyTwoWeekLow"))],["Dividend Yield",fmt_pct(safe_float(info.get("dividendYield"))*100)]
        ],columns=["Metric","Value"])
        st.dataframe(fund,use_container_width=True,hide_index=True)
        st.markdown("#### Earnings + Momentum + Unusual Volume")
        x=add_indicators(df).iloc[-1] if not df.empty else pd.Series(dtype=float)
        m=st.columns(5)
        for col,label,key in [(m[0],"1D","Ret1D"),(m[1],"5D","Ret5D"),(m[2],"20D","Ret20D")]: col.metric(label,fmt_pct(x.get(key,np.nan)))
        m[3].metric("Rel Volume",f"{safe_float(x.get('RelVol')):.2f}x" if np.isfinite(safe_float(x.get('RelVol'))) else "—")
        m[4].metric("RSI",fmt_num(x.get("RSI")))
        if np.isfinite(safe_float(x.get("RelVol"))) and safe_float(x.get("RelVol"))>=2: st.success("Unusual volume detected: at least 2× the 20-day average.")
        st.markdown("</div>",unsafe_allow_html=True)
    with right: render_news_rail(f"{symbol} News",symbol)

@st.cache_data(ttl=3600, show_spinner=False)
def analyst_forecast_detail(ticker: str) -> pd.DataFrame:
    rows=[]
    try:
        rec=getattr(yf.Ticker(ticker),"recommendations",None)
        if isinstance(rec,pd.DataFrame) and not rec.empty:
            x=rec.reset_index()
            for _,r in x.tail(100).iterrows():
                period=str(r.get("period",r.get("Period","")))
                firm=r.get("firm",r.get("Firm",""))
                to_grade=r.get("toGrade",r.get("To Grade",r.get("to_grade","")))
                from_grade=r.get("fromGrade",r.get("From Grade",r.get("from_grade","")))
                action=r.get("action",r.get("Action",""))
                rows.append({"Date":r.get("Date",r.get("date",r.get("index",""))),"Analyst/Firm":firm,"Rating":to_grade,"Previous":from_grade,"Action":action,"Period":period})
    except Exception:
        pass
    if BENZINGA_KEY:
        b=analyst_benzinga(ticker)
        if not b.empty:
            for _,r in b.iterrows():
                rows.append({"Date":r.get("date",""),"Analyst/Firm":r.get("firm",r.get("analyst","")),"Rating":r.get("rating_current",r.get("rating","")),"Previous":r.get("rating_prior",""),"Action":r.get("action_company",r.get("action","")),"Period":"12M","Target":r.get("pt_current",np.nan)})
    out=pd.DataFrame(rows)
    if out.empty: return out
    if "Date" in out.columns: out["Date"]=pd.to_datetime(out["Date"],errors="coerce").dt.date
    cutoff=date.today()-timedelta(days=365)
    if "Date" in out.columns:
        out=out[(out["Date"].isna()) | (out["Date"]>=cutoff)]
    return out.drop_duplicates().sort_values("Date",ascending=False,na_position="last")

# ============================================================
# Analyst
# ============================================================
with T[7]:
    left,right=page_columns()
    with left:
        st.markdown(f"<div class='card-title'>⭐ {symbol} Analyst Ratings & Target Price</div>",unsafe_allow_html=True)
        current=safe_float(info.get("currentPrice"))
        low=safe_float(info.get("targetLowPrice")); avg=safe_float(info.get("targetMeanPrice")); high=safe_float(info.get("targetHighPrice"))
        top=st.columns(4); top[0].metric("Current",f"${current:.2f}" if np.isfinite(current) else "—"); top[1].metric("Target Low",f"${low:.2f}" if np.isfinite(low) else "—"); top[2].metric("Target Avg",f"${avg:.2f}" if np.isfinite(avg) else "—"); top[3].metric("Target High",f"${high:.2f}" if np.isfinite(high) else "—")
        if np.isfinite(current) and np.isfinite(avg): st.metric("Average target upside",fmt_pct((avg/current-1)*100))
        rec_key=info.get("recommendationKey",info.get("recommendationMean","—"))
        st.markdown("<div class='forecast-shell'><div class='forecast-title'>12-Month Analyst Outlook</div><div class='forecast-sub'>Consensus target range and current quantitative decision signal</div>",unsafe_allow_html=True)
        gc1,gc2=st.columns([1,1.55],gap="medium")
        with gc1: st.plotly_chart(decision_forecast_chart(score.get("score",50),score.get("label","Hold"),current,low,avg,high),use_container_width=True,key="v8_decision_gauge")
        with gc2: st.plotly_chart(target_range_chart(current,low,avg,high),use_container_width=True,key="v8_target_range")
        st.markdown(f"<div class='hint'><b>Analyst rating:</b> {rec_key} &nbsp; • &nbsp; <b>Low:</b> {fmt_money(low)} &nbsp; • &nbsp; <b>Average:</b> {fmt_money(avg)} &nbsp; • &nbsp; <b>High:</b> {fmt_money(high)}</div></div>",unsafe_allow_html=True)
        detail=analyst_forecast_detail(symbol)
        st.markdown("#### Detailed analyst forecast / rating history — last 12 months")
        if detail.empty:
            detail=pd.DataFrame([{"Date":"—","Analyst/Firm":"Consensus","Analyst Rating":rec_key,"Target":avg,"12M Low":low,"12M Avg":avg,"12M High":high,"Previous":"—","Action":"Consensus","Period":"12M"}])
        else:
            if "Target" in detail.columns: detail["Target"] = pd.to_numeric(detail["Target"],errors="coerce")
            detail["12M Low"] = low; detail["12M Avg"] = avg; detail["12M High"] = high
            detail["Analyst Rating"] = detail.get("Rating", "")
        show=[c for c in ["Date","Analyst/Firm","Analyst Rating","Target","12M Low","12M Avg","12M High","Previous","Action","Period"] if c in detail.columns]
        st.dataframe(detail[show],use_container_width=True,hide_index=True)
        ratings=analyst_benzinga(symbol)
        if not ratings.empty:
            cols=[c for c in ["date","analyst","firm","rating_current","pt_current","action_company","action_pt","importance"] if c in ratings.columns]
            st.dataframe(ratings[cols],use_container_width=True,hide_index=True)
        st.markdown("#### Earnings context")
        eh = earnings_yahoo([symbol])
        if not eh.empty:
            st.dataframe(normalize_earnings(eh), use_container_width=True, hide_index=True)
        else:
            st.caption("Earnings date unavailable")
    with right: render_news_rail("Analyst / Research News",symbol)

# ============================================================
# Earnings
# ============================================================
with T[8]:
    left,right=page_columns()
    with left:
        st.markdown("<div class='card-title'>📅 Stocks Earnings Calendar</div>",unsafe_allow_html=True)
        days={"Next 7 days":7,"Next 14 days":14,"Next 30 days":30}[earnings_window]
        ed=normalize_earnings(earnings_events(days))
        if not ed.empty:
            for i,row in ed.head(50).iterrows():
                t=str(row.get("Ticker","")); dt=str(row.get("Earnings Date",""))[:10]
                clickable_ticker(t,f"earn_list_{i}",dt)
                extras=[]
                for c in ["EPS Est","EPS Actual","Time"]:
                    if c in row and pd.notna(row[c]): extras.append(f"{c}: {row[c]}")
                if extras: st.caption(" • ".join(extras))
        st.markdown("<div class='card'><div class='card-title'>📚 Earnings History — {}</div>".format(symbol),unsafe_allow_html=True)
        try:
            hist=yf.Ticker(symbol).earnings_history
            if isinstance(hist,pd.DataFrame) and not hist.empty: st.dataframe(hist.tail(12).reset_index(),use_container_width=True,hide_index=True)
        except Exception: pass
        st.markdown("</div>",unsafe_allow_html=True)
    with right: render_news_rail("Earnings Stories",symbol)

# ============================================================
# Technical + Risk
# ============================================================
with T[9]:
    left,right=page_columns()
    with left:
        st.markdown(f"<div class='card-title'>📊 {symbol} Stock Analysis — Decision + 12-Month Forecast</div>",unsafe_allow_html=True)
        if not df.empty:
            d=add_indicators(df); x=d.iloc[-1]; price=safe_float(x["Close"]); atr=safe_float(x["ATR14"]); atr=atr if np.isfinite(atr) else price*.03
            decision = "BUY" if safe_float(score.get("score")) >= 72 else "HOLD" if safe_float(score.get("score")) >= 45 else "SELL"
            current=safe_float(info.get("currentPrice",price)); low=safe_float(info.get("targetLowPrice")); avg=safe_float(info.get("targetMeanPrice")); high=safe_float(info.get("targetHighPrice"))
            st.markdown("<div class='forecast-shell'><div class='forecast-title'>Decision Matrix</div><div class='forecast-sub'>Buy / Hold / Sell positioning from the quantitative score, paired with the next 12-month analyst target range.</div>",unsafe_allow_html=True)
            g1,g2=st.columns([1,1.55],gap="medium")
            with g1: st.plotly_chart(decision_forecast_chart(score.get("score",50),decision,current,low,avg,high),use_container_width=True,key="v8_stock_decision")
            with g2: st.plotly_chart(target_range_chart(current,low,avg,high),use_container_width=True,key="v8_stock_target_range")
            st.markdown(f"<div class='hint'><b>{decision}</b> • Quant score {int(safe_float(score.get('score',0))):d}/100 &nbsp; • &nbsp; Current {fmt_money(current)} &nbsp; • &nbsp; 12M low {fmt_money(low)} &nbsp; • &nbsp; avg {fmt_money(avg)} &nbsp; • &nbsp; high {fmt_money(high)}</div></div>",unsafe_allow_html=True)
            st.plotly_chart(price_chart(df,symbol),use_container_width=True,key="v5_tech_price")
            a,b=st.columns(2)
            with a:
                rf=go.Figure(go.Scatter(x=d.index,y=d["RSI"],name="RSI",line=dict(color="#1687f8",width=2))); rf.add_hline(y=70,line_dash="dot"); rf.add_hline(y=30,line_dash="dot"); rf.update_layout(template="plotly_white",height=260,margin=dict(l=0,r=0,t=20,b=0),paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(rf,use_container_width=True,key="v5_rsi")
            with b:
                mf=go.Figure(); mf.add_trace(go.Scatter(x=d.index,y=d["MACD"],name="MACD")); mf.add_trace(go.Scatter(x=d.index,y=d["MACDSignal"],name="Signal")); mf.update_layout(template="plotly_white",height=260,margin=dict(l=0,r=0,t=20,b=0),paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(mf,use_container_width=True,key="v5_macd")
            stop=price-1.5*atr; target1=price+2*atr; target2=price+3*atr; rr=(target1-price)/(price-stop) if price>stop else np.nan
            st.markdown("#### Risk / Reward")
            q=st.columns(5); q[0].metric("ATR14",f"${atr:.2f}"); q[1].metric("Stop",f"${stop:.2f}"); q[2].metric("Target 1",f"${target1:.2f}"); q[3].metric("Target 2",f"${target2:.2f}"); q[4].metric("R:R",f"{rr:.2f}x" if np.isfinite(rr) else "—")
            st.caption("Illustrative analytical levels only; not personalized investment advice.")
        else: st.warning("No technical data available.")
    with right: render_news_rail("Technical / Catalyst News",symbol)

# ============================================================
# Options
# ============================================================
with T[10]:
    left,right=page_columns()
    with left:
        st.markdown(f"<div class='card-title'>🎯 {symbol} Stock Options</div>",unsafe_allow_html=True)
        calls,puts,op=options_data(symbol)
        if not op: st.warning("Options data unavailable for this ticker or Yahoo did not return an option chain.")
        else:
            q=st.columns(3); q[0].metric("Nearest expiry",op["expiry"]); q[1].metric("Put/Call OI",fmt_num(op["put_call_oi"])); q[2].metric("Put/Call Volume",fmt_num(op["put_call_volume"]))
            pc=safe_float(op["put_call_volume"])
            if np.isfinite(pc) and pc<.7: st.success("Relatively call-heavy options activity.")
            elif np.isfinite(pc) and pc>1.3: st.error("Relatively put-heavy options activity.")
            else: st.info("Options activity is relatively balanced.")
            c1,c2=st.columns(2)
            with c1: st.markdown("#### Highest OI Calls"); st.dataframe(calls.sort_values("openInterest",ascending=False).head(20),use_container_width=True,hide_index=True)
            with c2: st.markdown("#### Highest OI Puts"); st.dataframe(puts.sort_values("openInterest",ascending=False).head(20),use_container_width=True,hide_index=True)
    with right: render_news_rail("Options / Volatility News",symbol)

# ============================================================
# IPOs
# ============================================================
with T[11]:
    left,right=page_columns()
    with left:
        st.markdown("<div class='card-title'>🚀 Stocks / IPOs / New Listings Analysis</div>",unsafe_allow_html=True)
        if BENZINGA_KEY:
            data=benzinga_get("calendar/ipos",{"pagesize":100}); rows=(data or {}).get("ipos",[]) if isinstance(data,dict) else []
            if not rows: rows=(data or {}).get("data",[]) if isinstance(data,dict) else []
            idf=pd.DataFrame(rows)
            if idf.empty: st.info("Benzinga IPO endpoint returned no rows right now.")
            else:
                if "ticker" in idf.columns:
                    for i,r in idf.head(30).iterrows(): clickable_ticker(str(r.get("ticker")),f"ipo_{i}",str(r.get("pricing_date",r.get("date",""))))
                st.dataframe(idf,use_container_width=True,hide_index=True)
        else:
            pass
    with right: render_news_rail("IPO / New Listing Stories")

# ============================================================
# Stock Education
# ============================================================
with T[12]:
    st.markdown("<div class='hero-card'><div class='hero-ticker'>🎓 Stock Trading Academy</div><div class='hero-company'>A structured learning center covering stocks, charts, fundamentals, options, risk management and practical trade planning. Educational content only.</div></div>",unsafe_allow_html=True)

    st.markdown("### 1. Stock market foundations")
    basics=st.columns(4,gap="small")
    cards=[
        ("Price & spread","Bid is the highest current buyer price. Ask is the lowest seller price. The bid-ask spread is the cost of crossing between buyers and sellers."),
        ("Market cap","Market capitalization = share price × shares outstanding. Large caps are generally more established; small caps can move more sharply."),
        ("Liquidity","High liquidity usually means tighter spreads and easier entries/exits. Volume and average dollar volume help measure tradability."),
        ("Volatility","Volatility measures how widely price moves. ATR and implied volatility are two useful ways to understand expected movement."),
    ]
    for c,(title,desc) in zip(basics,cards):
        with c: st.markdown(f"<div class='edu-card'><div class='edu-term'>{title}</div><div class='edu-def'>{desc}</div></div>",unsafe_allow_html=True)

    st.markdown("### 2. How to read a stock chart")
    c1,c2=st.columns([1.25,.75],gap="medium")
    with c1:
        fig=go.Figure()
        x=list(range(1,13)); o=[100,103,101,106,108,105,110,112,111,115,118,116]; h=[104,106,105,110,112,111,114,116,115,120,121,120]; l=[98,100,99,103,105,102,107,109,108,112,115,113]; c=[103,101,104,108,106,110,113,111,115,118,116,119]
        fig.add_trace(go.Candlestick(x=x,open=o,high=h,low=l,close=c,name="Price"))
        ma=[np.mean(c[max(0,i-2):i+1]) for i in range(len(c))]
        fig.add_trace(go.Scatter(x=x,y=ma,name="3-period moving average",line=dict(width=2)))
        fig.update_layout(template="plotly_white",height=330,xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=15,b=0),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True,key="edu_candles_v82")
    with c2:
        st.markdown("**Candlestick anatomy**")
        st.markdown("""- **Body:** open-to-close price range.
- **Wick:** intraday high/low beyond the body.
- **Green/up candle:** close above open.
- **Red/down candle:** close below open.
- **Support:** area where buying has historically appeared.
- **Resistance:** area where selling has historically appeared.
- **Trend:** sequence of higher highs/lows or lower highs/lows.""")

    st.markdown("### 3. Technical indicators — what each one answers")
    ind=st.columns(3,gap="small")
    indicator_cards=[
        ("SMA / EMA","Is the price trend rising or falling? EMA reacts faster to recent prices."),("RSI","Is momentum becoming stretched? RSI near 70/30 is context, not an automatic sell/buy signal."),("MACD","Is momentum/trend changing? Watch the MACD line, signal line and histogram."),("Bollinger Bands","Is volatility expanding or contracting, and where is price relative to its moving-average envelope?"),("ATR","How much does the stock typically move? Useful for position sizing and risk levels."),("Relative Volume","Is today's trading activity unusually high versus recent average volume?")]
    for c,(title,desc) in zip(ind,indicator_cards):
        with c: st.markdown(f"<div class='edu-card'><div class='edu-term'>{title}</div><div class='edu-def'>{desc}</div></div>",unsafe_allow_html=True)

    st.markdown("### 4. Fundamental analysis")
    f1,f2=st.columns(2,gap="medium")
    with f1:
        st.markdown("**Growth & profitability**")
        st.markdown("Revenue growth → EPS growth → gross margin → operating margin → free cash flow → return on invested capital. Look for consistency and improving economics rather than one isolated quarter.")
    with f2:
        st.markdown("**Valuation**")
        st.markdown("P/E compares price with earnings. Forward P/E uses expected earnings. PEG relates valuation to growth. EV/EBITDA is useful when comparing businesses with different capital structures. Always compare valuation with peers and growth.")

    st.markdown("### 5. Order types")
    orders=st.columns(4)
    order_data=[("Market","Prioritizes execution speed; final price can vary."),("Limit","Executes only at your limit price or better."),("Stop","Becomes a market order after the trigger is reached."),("Stop-limit","Adds a limit price, reducing price uncertainty but risking no fill.")]
    for c,(t,d) in zip(orders,order_data):
        with c: st.markdown(f"<div class='edu-card'><div class='edu-term'>{t}</div><div class='edu-def'>{d}</div></div>",unsafe_allow_html=True)

    st.markdown("### 6. Options trading — complete beginner map")
    st.markdown("**Call = right to buy. Put = right to sell.** Every option has an underlying stock, strike price, expiration and premium. A standard equity option generally represents 100 shares.")
    opt=st.columns(3,gap="small")
    opt_data=[("ITM / ATM / OTM","In-the-money has intrinsic value; at-the-money is near the strike; out-of-the-money has no intrinsic value."),("Premium","The option price paid by the buyer and received by the seller."),("Expiration","The contract's final date. Time value generally declines as expiration approaches, all else equal."),("Implied Volatility","Market-implied expectation of future movement. Higher IV usually means more expensive options."),("Open Interest","Number of outstanding contracts. Volume measures today's trading activity."),("Assignment","Short option sellers can be assigned. Understand exercise/assignment mechanics before selling options.")]
    for c,(t,d) in zip(opt,opt_data):
        with c: st.markdown(f"<div class='edu-card'><div class='edu-term'>{t}</div><div class='edu-def'>{d}</div></div>",unsafe_allow_html=True)

    st.markdown("### 7. Options Greeks")
    greek=st.columns(5)
    for c,t,d in zip(greek,["Delta","Gamma","Theta","Vega","Rho"],["Price sensitivity to the stock","How quickly delta changes","Time decay","Sensitivity to implied volatility","Sensitivity to interest rates"]):
        with c: st.markdown(f"<div class='edu-card'><div class='edu-term'>{t}</div><div class='edu-def'>{d}</div></div>",unsafe_allow_html=True)

    st.markdown("### 8. Long option payoff — visual example")
    strike=100; premium=5; spot=np.linspace(60,140,161); call=np.maximum(spot-strike,0)-premium; put=np.maximum(strike-spot,0)-premium
    pf=go.Figure(); pf.add_trace(go.Scatter(x=spot,y=call,name="Long Call")); pf.add_trace(go.Scatter(x=spot,y=put,name="Long Put")); pf.add_hline(y=0,line_dash="dot"); pf.add_vline(x=strike,line_dash="dot",annotation_text="Strike $100"); pf.update_layout(template="plotly_white",height=330,xaxis_title="Stock price at expiration",yaxis_title="Profit / Loss per share",margin=dict(l=0,r=0,t=20,b=0),paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(pf,use_container_width=True,key="edu_option_payoff_v82")

    st.markdown("### 9. A disciplined stock-research workflow")
    flow=go.Figure(); nodes=[("1","Market regime"),("2","Sector"),("3","Trend"),("4","Fundamentals"),("5","Catalyst"),("6","Volume / Options"),("7","Risk"),("8","Plan")]; xs=list(range(len(nodes)))
    for x,(n,label) in zip(xs,nodes):
        flow.add_shape(type="rect",x0=x-.38,x1=x+.38,y0=-.25,y1=.25,line=dict(color="#1687f8"),fillcolor="#eef7ff"); flow.add_annotation(x=x,y=0,text=f"<b>{n}</b><br>{label}",showarrow=False,font=dict(size=10,color="#173f69"));
        if x<len(nodes)-1: flow.add_annotation(x=x+.5,y=0,text="→",showarrow=False,font=dict(size=18,color="#1687f8"))
    flow.update_xaxes(visible=False,range=[-.8,len(nodes)-.2]); flow.update_yaxes(visible=False,range=[-.6,.6]); flow.update_layout(height=160,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(flow,use_container_width=True,key="edu_workflow_v82")

    st.markdown("### 10. Risk management checklist")
    st.markdown("Before entering a trade, define: **thesis → entry → invalidation → position size → maximum acceptable loss → catalyst/event risk → exit plan**. Never treat a technical score or AI label as a guarantee of future performance.")
    r1,r2,r3,r4=st.columns(4); r1.metric("Position sizing","Size for risk"); r2.metric("Invalidation","Define before entry"); r3.metric("Options","Know max loss"); r4.metric("Events","Check earnings")

# ============================================================
# Activity / stock logs
# ============================================================
with T[13]:
    st.markdown("<div class='card-title'>🧾 Stock Activity & Lookup Log</div>",unsafe_allow_html=True)
    st.caption("This records which symbols were searched/opened during this browser session and also writes technical events to logs/app.log when the app has filesystem access.")
    if st.session_state.activity_log:
        logdf=pd.DataFrame(st.session_state.activity_log[::-1])
        st.dataframe(logdf,use_container_width=True,hide_index=True)
        if st.button("Clear session log",key="clear_activity"):
            st.session_state.activity_log=[]; st.rerun()
    else:
        st.info("No stock lookups yet. Search a ticker or click a stock button to build the log.")
    st.markdown("### Current stock context")
    ci=st.columns(6); ci[0].metric("Symbol",symbol); ci[1].metric("Company",str(info.get("shortName","—"))[:22]); ci[2].metric("Price",fmt_money(score.get("price"))); ci[3].metric("1D",fmt_pct(score.get("momentum"))); ci[4].metric("Signal",score.get("label","—")); ci[5].metric("Score",f"{score.get('score',0)}/100")

if st.session_state.stock_detail_open:
    stock_detail_dialog(st.session_state.selected_symbol)

st.markdown(f"<div class='footer'>AI Stock Analyzer • Market data via yfinance • Auto-refresh {'enabled' if auto_refresh else 'disabled'} • Last render {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",unsafe_allow_html=True)
