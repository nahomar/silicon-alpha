# market-pattern-bot

Scrapes Reddit, X/Twitter, and financial news; scores sentiment; pulls price
data; runs correlation, clustering, anomaly screening, and ticker-level
lead/lag analysis between social sentiment and returns; writes a Markdown
report.

## Install
```bash
cd ~/market-pattern-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys (all optional, fallbacks exist)
```

## Run
```bash
python main.py              # full pipeline
python main.py --quick      # smaller sample
python main.py --no-scrape  # reuse last scrape, re-run analysis only
```

Output:
- `reports/raw/*.json` – raw scraped items per run
- `reports/raw/history.parquet` – rolling history across runs
- `reports/report_YYYYMMDDTHHMMSS.md` – Markdown analysis

## Layout
```
scrapers/  reddit_scraper.py  x_scraper.py  news_scraper.py
data/      market_data.py     storage.py
ml/        sentiment.py       correlation.py  patterns.py
analysis/  report.py
main.py
```

## Sources used

### Fallback strategy per source
- **Reddit**: PRAW → public JSON (`/r/<sub>/hot.json`)
- **X**: API v2 bearer → snscrape → Nitter RSS mirror
- **News**: NewsAPI → Google News RSS → static RSS (SEC, Fed, MarketWatch, CNBC, Reuters)

### Real-world context packed into the report
Regulatory & market events used as the April 17–20, 2026 backdrop are pulled
from the Federal Register, SEC press, Fed calendar, Bloomberg, CNBC, Zacks
coverage, and the ECMI / IMF outlook. See `analysis/report.py`.

## What the ML actually does
1. **VADER sentiment** per scraped item (swap in FinBERT trivially).
2. **Ticker extraction** via regex with a heavy stopword list.
3. **Correlation matrix** across indexes / sector ETFs / megacaps on 6-month
   daily returns; pairs with |ρ| ≥ 0.8 surfaced.
4. **KMeans clustering** of tickers by return vectors.
5. **Anomaly screen** – today's z-score vs trailing 63-day mean/std.
6. **Lead-lag cross-correlation** of daily mean sentiment vs daily returns,
   ±5 days. Positive best-lag means sentiment leads price.
7. **Mention-spike detector** – z-score of today's ticker mention count vs
   14-day rolling mean.
```
