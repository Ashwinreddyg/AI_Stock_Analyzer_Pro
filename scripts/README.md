# Automatic Daily Scan

The application includes a standalone `scripts/daily_scan.py` so the daily scan does not depend on a Streamlit browser session.

## Windows Task Scheduler
From the project root:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
python .\scripts\daily_scan.py
```

Schedule `python.exe` and pass `scripts\daily_scan.py` as the argument. A weekday schedule around 9:35 AM Eastern is a practical starting point.

## GitHub Actions
The included workflow runs on weekdays and stores the report plus recommendation history in the repository. Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` under repository Settings → Secrets and variables → Actions. Use `workflow_dispatch` to test manually.

## Cost / data note
The quantitative layer uses the free unofficial `yfinance` package and public Yahoo endpoints; those endpoints can change or rate-limit. LLM APIs can incur usage charges. For a zero-API-cost LLM, use a local Ollama installation rather than cloud LLM APIs.
