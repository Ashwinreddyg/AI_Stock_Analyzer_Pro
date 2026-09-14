# AI Stock Analyzer Pro v8.3 — LLM Market Intelligence

A polished light-blue Streamlit stock research dashboard that combines a free yfinance quantitative scanner with an optional Claude/OpenAI/Local-Ollama explanation layer.

## What's new in v8

### 1. AI/LLM-powered Market Scanner
- Quantitative engine scans 100/250/350+ symbols using yfinance/Yahoo public endpoints.
- Technical + fundamental + earnings + news sentiment + unusual volume + options + market-regime score.
- Produces Top 5/10 BUY, WATCH and AVOID candidates.
- Claude or OpenAI explains why each candidate received its decision, compares catalysts, summarizes supplied headlines, and highlights risks.
- Optional Local Ollama provider can make the explanation layer local/free if you run a model on your own PC.
- Daily Top 10 report is generated once per day/provider/model combination and can be downloaded as Markdown.
- The app also writes `data/daily_reports/stock_report_YYYY-MM-DD.md` when the filesystem is persistent.

### 2. Editable stock search
- Sidebar search is editable and is the single stock lookup control.
- The duplicate global header search/status controls were removed for a cleaner terminal layout.
- Clicking any ticker in the app updates the active stock.

### 3. Market strip upgraded
SPY, QQQ, DIA, IWM, VIXY, GLD, USO and TLT now appear in one horizontal row with price, today's % change and a compact 1-month sparkline.

### 4. Analyst forecast and decision view
- 12-month low / average / high target summary.
- Current analyst rating/recommendation from Yahoo when available.
- Detailed analyst rating history for the last 12 months when available.
- A visual BUY / HOLD / SELL decision gauge plus current price and 12-month analyst target range.
- Consensus is shown cleanly even when firm-level rows are unavailable; no API-key warning clutter is displayed.

### 5. Stock Education tab
Expanded learning center covering:
- market structure, bid/ask/spread, liquidity and volatility
- candlesticks, support/resistance and trend reading
- SMA/EMA, RSI, MACD, ATR, Bollinger Bands and relative volume
- revenue, EPS, margins, free cash flow and valuation ratios
- market, limit, stop and stop-limit orders
- calls/puts, strike, premium, expiration, ITM/ATM/OTM and assignment
- Delta, Gamma, Theta, Vega and Rho
- long call/long put payoff graphics
- a complete research workflow and risk-management checklist

### 6. Cleaner interface
- Removed duplicate header search/status controls.
- Removed the sidebar Layout toggles to reduce clutter.
- Added more chart intervals: 5m, 15m, 30m, 1h, 1d, 1wk and 1mo.
- Intraday intervals are automatically constrained to Yahoo's practical retention windows.

### 7. Stock activity logs
- Session activity log records searched/opened tickers.
- `logs/app.log` records application events when filesystem access is available.
- Activity Log tab shows the current browser session's stock lookup history.

## Important cost/data note
`yfinance` is free and does not require a Yahoo API key, but it is an unofficial/community library that uses public Yahoo Finance endpoints. Yahoo endpoints can change or rate-limit requests, and this is not a guaranteed licensed commercial market-data feed.

The **LLM explanation layer is not inherently free** when using OpenAI or Anthropic API keys. If you want a no-API-cost LLM layer, run Local Ollama on your own machine. API usage can also be disabled; the quantitative scanner still works.

The BUY/WATCH/AVOID output is research/education, not personalized investment advice.

## Windows setup

```powershell
cd C:\Users\ashwi\Downloads\AI_Stock_Analyzer_Pro_v8
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open: `http://localhost:8501`

### Optional API keys

PowerShell for the current session:

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_KEY"
$env:OPENAI_MODEL="gpt-5"
$env:ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
$env:CLAUDE_MODEL="claude-opus-5"
$env:BENZINGA_API_KEY="YOUR_BENZINGA_KEY"
```

Then restart Streamlit.

### Local Ollama

Install Ollama separately, download a local model, start Ollama, and select **Local Ollama** in the scanner. The default endpoint is `http://localhost:11434/api/generate` and the default model field is `llama3.2:3b`.

## Recommended central deployment

### Best free/easiest: Streamlit Community Cloud
1. Create a GitHub repository and push this folder.
2. Sign in to Streamlit Community Cloud.
3. Create an app and select `app.py`.
4. Add secrets in the deployment settings instead of committing API keys.
5. Share the resulting `streamlit.app` URL.

This is the easiest option for a personal/educational shared dashboard. The app can be public or private depending on the Community Cloud sharing settings.

### Best free-ish alternative: Render
The included `Dockerfile` and `render.yaml` can be used to deploy a free web service. Render's free web services spin down after inactivity and have ephemeral local storage, so do not rely on local `data/` or `logs/` for permanent history.

### Best scalable architecture: Google Cloud Run
Use the included `Dockerfile`. Cloud Run has a pay-per-use model and an always-free tier, subject to current limits. For a larger multi-user deployment, pair it with a persistent database/object store and move scheduled scans into a separate job.

## Recommended production architecture

For a truly central, multi-user version:

**Browser → Streamlit/Cloud Run → Scanner service → yfinance/Benzinga → LLM service → PostgreSQL → daily report store**

Add a scheduled worker (GitHub Actions, Cloud Run Job, or another scheduler) to run the scan once before market open and once after close. Store scan results in PostgreSQL rather than the local filesystem. This makes daily history, leaderboards, user watchlists and audit logs persistent.

## High-value next improvements

1. Persistent PostgreSQL scan history and per-symbol score history.
2. Pre-market / regular-hours / after-hours scanner modes.
3. Sector-relative scoring and benchmark-relative momentum.
4. News-source diversity and source timestamps.
5. Analyst target-change detection and consensus dispersion.
6. Options IV percentile, unusual options volume, max-pain and open-interest concentration.
7. Backtesting of BUY/WATCH/AVOID thresholds.
8. User accounts + saved watchlists.
9. Email/Telegram/Discord daily Top 10 delivery.
10. LLM confidence + source citations, with strict grounding to retrieved data.
11. Scheduled daily report generation independent of a browser session.
12. Persistent audit trail showing exactly which data points produced every score.

## v8 AI Daily Intelligence Center

v8 adds a professional research-terminal workflow around the quantitative scanner:

- Morning market briefing generated from supplied market data
- AI Top 10 BUY / WATCH / AVOID rankings
- "Why this stock today?" and "What could make this recommendation wrong?" reasoning
- Biggest analyst upgrades / downgrades when Benzinga is configured
- Earnings catalysts
- Unusual options activity
- Unusual volume
- News catalysts and sentiment
- Sector rotation dashboard
- Market regime
- Historical performance tracking for saved recommendations
- Daily report archive under `data/daily_reports/`
- `scripts/daily_scan.py` for unattended execution
- `.github/workflows/daily_scan.yml` for scheduled weekday GitHub Actions runs

### Recommended centralized deployment

**Free/easiest:** GitHub + Streamlit Community Cloud for the web UI, with GitHub Actions for the unattended daily scan. Store API keys in Streamlit Cloud secrets and GitHub Actions secrets.

**More production-oriented:** Docker/Render or Google Cloud Run for the UI, plus a persistent database/object store for recommendation history and reports. Do not depend on the local filesystem for persistent data on ephemeral cloud services.

### Daily scheduling

The Streamlit page is not a reliable background scheduler. Use `scripts/daily_scan.py` with Windows Task Scheduler, Linux cron, GitHub Actions, or a cloud scheduler. The included GitHub workflow runs on weekdays and commits generated reports/history back to the repository.

### AI providers

- OpenAI: cloud LLM explanation layer
- Claude: cloud LLM explanation layer
- Local Ollama: local/private LLM option
- No LLM configured: the app still provides the quantitative ranking and deterministic briefing

The yfinance layer is unofficial and uses public Yahoo Finance endpoints. It does not require a Yahoo API key, but endpoints can change or rate-limit. Cloud LLM APIs may incur charges.

## v8.5 deployment for personal/learning use

### Recommended: Streamlit Community Cloud (free)
1. Create/sign in to GitHub.
2. Create a repository, for example `ai-stock-analyzer`.
3. Upload the contents of this folder so `app.py` and `requirements.txt` are at the repository root.
4. Go to Streamlit Community Cloud and connect GitHub.
5. Choose **Deploy an app**, select your repository and `app.py`.
6. Deploy. Streamlit provides an `*.streamlit.app` URL.
7. If using OpenAI/Claude, open the app's Settings/Secrets area and add the required secrets (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and optional model names). Never commit API keys to GitHub.
8. Push future code changes to GitHub; Community Cloud can automatically update the deployed app.

### Alternative: Render Free
1. Push the project to GitHub.
2. Create a Render account and choose **New > Web Service**.
3. Connect the GitHub repository.
4. Use the included `render.yaml`/Dockerfile, or set the start command to `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`.
5. Choose the free service type where available.
6. Add API keys as environment variables in Render, not in source code.
7. Deploy and use the generated HTTPS URL.

### Alternative: Google Cloud Run
Use this when you want to learn containers/cloud deployment. Build the included Dockerfile and deploy the container to Cloud Run. Cloud Run has a monthly free tier, subject to usage limits; billing setup may still be required. This is more involved than Streamlit Community Cloud.

### Best choice for this project
For a personal/learning application, start with **Streamlit Community Cloud**. It is the simplest path from GitHub to a shareable Streamlit URL. Move to Cloud Run later when you need scheduled jobs, persistent storage, stronger control, or a more production-like architecture.


## v8.5 UI / interaction updates

- Market instruments **SPY, QQQ, DIA, IWM, VIXY, GLD, USO and TLT** are now clickable and each market quote + mini chart is contained in one visual card.
- Stock tickers in gainers, losers, earnings, stories, AI scanner, daily intelligence, IPOs, comparison and other ticker-driven sections use a common clickable interaction.
- Ticker selection updates the active stock used by the analysis tabs.
- Standardized typography across the application; removed viewport-based oversized font scaling.
- Responsive sizing remains for narrow screens without making desktop text excessively large.
