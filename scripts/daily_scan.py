import os, json
from datetime import date, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
try:
    from openai import OpenAI
except Exception: OpenAI=None
try:
    from anthropic import Anthropic
except Exception: Anthropic=None

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; REPORTS=DATA/"daily_reports"; REPORTS.mkdir(parents=True,exist_ok=True)
HISTORY=DATA/"recommendation_history.csv"
OPENAI_KEY=os.getenv("OPENAI_API_KEY","").strip(); ANTHROPIC_KEY=os.getenv("ANTHROPIC_API_KEY","").strip()
OPENAI_MODEL=os.getenv("OPENAI_MODEL","gpt-5"); CLAUDE_MODEL=os.getenv("CLAUDE_MODEL","claude-opus-5")
UNIVERSE=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'AVGO', 'TSLA', 'ORCL', 'AMD', 'NFLX', 'CRM', 'PLTR', 'MU', 'TSM', 'QCOM', 'INTC', 'SMCI', 'SOFI', 'RIVN', 'TTD', 'GTLB', 'DUOL', 'OKLO', 'SOUN', 'BBAI', 'AI', 'PATH', 'FIVN', 'CALX', 'ADBE', 'CSCO', 'IBM', 'NOW', 'INTU', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'ADI', 'MELI', 'PANW', 'CRWD', 'SNOW', 'DDOG', 'NET', 'MDB', 'SHOP', 'UBER', 'ABNB', 'COIN', 'HOOD', 'PYPL', 'SQ', 'AFRM', 'NU', 'SOFI', 'IONQ', 'RKLB', 'ASTS', 'HIMS', 'TEM', 'CRSP', 'RXRX', 'BEAM', 'DNA', 'IONQ', 'ACHR', 'JOBY', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V', 'MA', 'COF', 'USB', 'PNC', 'TFC', 'BK', 'CME', 'ICE', 'SPGI', 'MCO', 'CB', 'AON', 'MMC', 'PGR', 'ALL', 'TRV', 'MET', 'PRU', 'UNH', 'LLY', 'JNJ', 'MRK', 'ABBV', 'PFE', 'BMY', 'AMGN', 'GILD', 'REGN', 'VRTX', 'ISRG', 'MDT', 'SYK', 'BSX', 'EW', 'DHR', 'TMO', 'ABT', 'ELV', 'CI', 'CVS', 'HCA', 'ZBH', 'BDX', 'IDXX', 'DXCM', 'RMD', 'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'PSX', 'VLO', 'HAL', 'DVN', 'FANG', 'KMI', 'WMB', 'OKE', 'ET', 'LNG', 'BKR', 'CTRA', 'APA', 'CAT', 'DE', 'GE', 'HON', 'RTX', 'BA', 'LMT', 'NOC', 'GD', 'UPS', 'FDX', 'UNP', 'CSX', 'WM', 'ETN', 'EMR', 'PH', 'ITW', 'MMM', 'CARR', 'JCI', 'TT', 'URI', 'FAST', 'PCAR', 'CMI', 'ROK', 'AME', 'VRSK', 'WMT', 'COST', 'TGT', 'HD', 'LOW', 'TJX', 'NKE', 'SBUX', 'MCD', 'CMG', 'YUM', 'KO', 'PEP', 'PM', 'MO', 'PG', 'CL', 'EL', 'KHC', 'MDLZ', 'GIS', 'K', 'HSY', 'MNST', 'KR', 'DG', 'DLTR', 'ORLY', 'AZO', 'ULTA', 'LULU', 'DECK', 'DHI', 'LEN', 'PHM', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'NFLX', 'WBD', 'SPOT', 'ROKU', 'FOXA', 'LYV', 'EA', 'TTWO', 'RBLX', 'U', 'MTCH', 'PINS', 'SNAP', 'RDDT', 'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'STLD', 'DOW', 'DD', 'ALB', 'VMC', 'MLM', 'CE', 'PPG', 'IFF', 'BALL', 'IP', 'AVY', 'MOS', 'CF', 'FMC', 'PLD', 'AMT', 'EQIX', 'CCI', 'O', 'SPG', 'DLR', 'PSA', 'WELL', 'VICI', 'AVB', 'EQR', 'ESS', 'ARE', 'CBRE', 'CSGP', 'IRM', 'ACN', 'CTSH', 'EPAM', 'GEN', 'HPE', 'HPQ', 'DELL', 'STX', 'WDC', 'ARM', 'MRVL', 'ON', 'NXPI', 'MCHP', 'MPWR', 'TER', 'SNPS', 'CDNS', 'ANSS', 'FTNT', 'ZS', 'OKTA', 'CHKP', 'AKAM', 'TTD', 'APP', 'GDDY', 'FSLY', 'ESTC', 'CFLT', 'TWLO', 'DOCU', 'ZM', 'VEEV', 'PAYC', 'HUBS', 'TEAM', 'WDAY', 'ASML', 'SAP', 'SONY', 'BABA', 'PDD', 'JD', 'BIDU', 'NTES', 'NIO', 'LI', 'XPEV', 'SE', 'MSTR', 'MARA', 'RIOT', 'CLSK', 'IBKR', 'RKT', 'HOPE', 'SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'TLT', 'XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLI', 'XLC', 'XLU', 'XLP', 'XLB', 'XLRE']
ETF_SET={"SPY","QQQ","DIA","IWM","GLD","TLT","XLF","XLK","XLE","XLV","XLY","XLI","XLC","XLU","XLP","XLB","XLRE"}
UNIVERSE=[x for x in dict.fromkeys(UNIVERSE) if x not in ETF_SET]

def sf(v):
    try:return float(v)
    except:return np.nan

def hist(t):
    try:
        d=yf.download(t,period="6mo",interval="1d",auto_adjust=False,progress=False,threads=False)
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        return d.dropna(how="all")
    except:return pd.DataFrame()

def indicators(d):
    d=d.copy(); c=pd.to_numeric(d["Close"],errors="coerce"); v=pd.to_numeric(d["Volume"],errors="coerce")
    for n in [21,50,200]: d[f"MA{n}"]=c.rolling(n).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rs=gain/loss.replace(0,np.nan); d["RSI"]=100-(100/(1+rs))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); d["MACD"]=e12-e26; d["MACDSignal"]=d["MACD"].ewm(span=9,adjust=False).mean(); d["RelVol"]=v/v.rolling(20).mean(); d["Ret20D"]=c.pct_change(20)*100
    return d

def tech(d):
    if d.empty or len(d)<60:return 50,np.nan,np.nan,np.nan,np.nan
    x=indicators(d).iloc[-1]; price=sf(x.get("Close")); s=50
    for ma,pts in [("MA21",5),("MA50",7),("MA200",7)]:
        q=sf(x.get(ma)); s += pts if np.isfinite(q) and price>q else -pts if np.isfinite(q) else 0
    r=sf(x.get("RSI"));
    if np.isfinite(r): s += 6 if 52<=r<=68 else 1 if 45<=r<52 else 2 if 68<r<=75 else -4 if r>78 else -5 if r<42 else 0
    s += 5 if sf(x.get("MACD"))>sf(x.get("MACDSignal")) else -5
    rv=sf(x.get("RelVol")); mom=sf(x.get("Ret20D")); s += 5 if np.isfinite(mom) and mom>5 else 3 if np.isfinite(mom) and mom>2 else -4 if np.isfinite(mom) and mom<-5 else 0
    return int(max(0,min(100,s))),price,mom,rv,r

def info(t):
    try:return yf.Ticker(t).info or {}
    except:return {}

def news(t):
    try:return yf.Ticker(t).news or []
    except:return []

def fund(i):
    s=50; pe=sf(i.get("forwardPE")); g=sf(i.get("revenueGrowth")); m=sf(i.get("profitMargins")); roe=sf(i.get("returnOnEquity"))
    if np.isfinite(pe):s+=8 if 0<pe<25 else 4 if pe<40 else -5 if pe>70 else 0
    if np.isfinite(g):s+=10 if g>.20 else 6 if g>.08 else -5 if g<0 else 0
    if np.isfinite(m):s+=6 if m>.15 else 2 if m>0 else -4
    if np.isfinite(roe):s+=5 if roe>.15 else 2 if roe>0 else -3
    return int(max(0,min(100,s)))

def sentiment(items):
    if not items:return 50
    pos=["beat","upgrade","buy","bullish","surge","rally","record","strong","growth","raised","approval","contract","deal","launch","outperform","ai demand"]
    neg=["miss","downgrade","sell","bearish","drop","falls","fell","weak","cut","warning","probe","lawsuit","offering","tariff","inflation","recall"]
    vals=[]
    for it in items[:8]:
        c=it.get("content",{}) if isinstance(it.get("content"),dict) else {}; title=str(c.get("title") or it.get("title") or "").lower(); p=sum(title.count(w) for w in pos); n=sum(title.count(w) for w in neg); vals.append(max(-100,min(100,(p-n)*18)))
    return int(max(0,min(100,50+np.mean(vals)/2)))

def earnings(t):
    s=50; note="No immediate earnings signal"
    try:
        cal=yf.Ticker(t).calendar
        if isinstance(cal,pd.DataFrame):cal=cal.to_dict()
        ds=cal.get("Earnings Date",[]) if isinstance(cal,dict) else []
        if not isinstance(ds,list):ds=[ds]
        if ds:
            dt=pd.to_datetime(ds[0],errors="coerce")
            if pd.notna(dt):
                days=(dt.date()-date.today()).days
                if 0<=days<=14:s+=12;note=f"Earnings in {days}d"
                elif 15<=days<=45:s+=5;note=f"Earnings in {days}d"
    except:pass
    return int(max(0,min(100,s))),note

def options(t):
    try:
        tk=yf.Ticker(t); ds=tk.options
        if not ds:return 50,"No options chain"
        ch=tk.option_chain(ds[0]); c=ch.calls; p=ch.puts; coi=pd.to_numeric(c.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0).sum(); poi=pd.to_numeric(p.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()
        if coi<=0:return 50,"No call OI"
        ratio=poi/coi; return (65,"Call-heavy") if ratio<.75 else (55,"Balanced") if ratio<=1.25 else (35,"Put-heavy")
    except:return 50,"Options unavailable"

def market_score():
    vals=[]
    for t in ["SPY","QQQ","IWM"]:
        d=hist(t)
        if not d.empty: vals.append(sf(indicators(d).iloc[-1].get("Ret20D")))
    vals=[x for x in vals if np.isfinite(x)]; r=float(np.mean(vals)) if vals else 0
    return int(max(0,min(100,50+r*4))),("Risk-On" if r>2 else "Neutral" if r>-2 else "Risk-Off")

def scan(n=250,enrich=30):
    ms,reg=market_score(); batch=UNIVERSE[:n]; raw=yf.download(tickers=batch,period="6mo",interval="1d",auto_adjust=False,progress=False,threads=True,group_by="column"); base=[]
    if isinstance(raw.columns,pd.MultiIndex):
        for t in batch:
            try:
                c=raw["Close"][t].dropna(); v=raw["Volume"][t].dropna() if "Volume" in raw else pd.Series(index=c.index,dtype=float)
                if len(c)<60:continue
                d=pd.DataFrame({"Close":c,"Volume":v}); ts,p,m,rv,r=tech(d); base.append((t,ts,p,m,rv,r))
            except:pass
    base=sorted(base,key=lambda x:(x[1],x[3] if np.isfinite(x[3]) else -999),reverse=True)[:enrich]; rows=[]
    for t,ts,p,m,rv,rsi in base:
        i=info(t); f=fund(i); nn=sentiment(news(t)); ee,en=earnings(t); oo,on=options(t); vol=70 if np.isfinite(rv) and rv>=2 else 60 if np.isfinite(rv) and rv>=1.5 else 50 if np.isfinite(rv) and rv>=1 else 40
        score=round(ts*.25+f*.20+ee*.15+nn*.15+vol*.10+oo*.05+ms*.10); dec="BUY" if score>=75 else "WATCH" if score>=55 else "AVOID"
        rows.append({"Ticker":t,"Decision":dec,"Score":score,"Technical":ts,"Fundamental":f,"Earnings":ee,"News":nn,"Volume":vol,"Options":oo,"Market":ms,"Price":p,"Momentum":m,"RelVol":rv,"RSI":rsi,"Earnings Signal":en,"Options Signal":on,"Market Regime":reg})
    return pd.DataFrame(rows).sort_values("Score",ascending=False).reset_index(drop=True)

def llm_report(df,reg):
    data=df.head(10).replace({np.nan:None}).to_dict("records")
    prompt="""You are a senior market-research editor. Use ONLY the supplied quantitative data. Never invent facts. Produce a concise morning report. For each Top 10 stock explain: why BUY/WATCH/AVOID, strongest confirming signal, and what could make the recommendation wrong. Include market regime, catalysts represented in the data, and risks. This is research, not personalized advice.\n\nMarket regime: %s\n\nDATA:\n%s"""%(reg,json.dumps(data,default=str,indent=2))
    try:
        if OPENAI_KEY and OpenAI:return OpenAI(api_key=OPENAI_KEY).responses.create(model=OPENAI_MODEL,input=prompt).output_text.strip()
        if ANTHROPIC_KEY and Anthropic:
            msg=Anthropic(api_key=ANTHROPIC_KEY).messages.create(model=CLAUDE_MODEL,max_tokens=5000,messages=[{"role":"user","content":prompt}]); return "\n".join(getattr(x,"text",str(x)) for x in msg.content)
    except Exception as e:return "LLM generation failed: %s"%e
    return "LLM not configured. Review the quantitative ranking below."

def main():
    n=int(os.getenv("SCAN_UNIVERSE","250")); enrich=int(os.getenv("SCAN_ENRICH","30")); df=scan(n,enrich)
    if df.empty:raise SystemExit("No scan results returned")
    hist_df=pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame(columns=["Date","Ticker","Decision","Score","Price"])
    add=df.head(30)[["Ticker","Decision","Score","Price"]].copy(); add.insert(0,"Date",date.today().isoformat()); hist_df=pd.concat([hist_df,add],ignore_index=True).drop_duplicates(["Date","Ticker"],keep="last"); hist_df.to_csv(HISTORY,index=False)
    _,reg=market_score(); report=llm_report(df,reg); path=REPORTS/("daily_intelligence_%s.md"%date.today().isoformat()); path.write_text("# AI Stock Analyzer Pro — Daily Intelligence\n\nGenerated: %s\n\n%s\n\n## Quantitative Top 10\n\n%s\n"%(datetime.now().isoformat(timespec="seconds"),report,df.head(10).to_markdown(index=False)),encoding="utf-8"); print(path)
if __name__=="__main__":main()
