# Signals and Systems

Data pipeline for dissertation research examining systematic risk and behavioral factor dominance through three interconnected essays. Aggregates financial, macroeconomic, and news sentiment data for culture war companies analysis.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd signalsandsystems

# Install dependencies
uv sync

# Copy environment template and fill in your API keys
cp .env.example .env
```

### Required API Keys

| Key | Purpose | Get it at |
|-----|---------|-----------|
| `FRED_API_KEY` | Federal Reserve economic data (inflation, rates, GDP, employment) | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `GUARDIAN_API_KEY` | Guardian news articles | [open-platform.theguardian.com](https://open-platform.theguardian.com/access/) |
| `NYT_API_KEY` / `NYT_API_SECRET` | New York Times articles | [developer.nytimes.com](https://developer.nytimes.com/) |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit posts and comments | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |
| `SEC_USER_AGENT` | SEC EDGAR Form 4 filings (name + email required by SEC) | N/A (use your own) |

## Usage

### Load all data

```python
from clean import get_clean_data

# Downloads (or loads from cache) and cleans all datasets
data = get_clean_data()
```

### Load specific datasets

```python
from clean import (
    load_comprehensive_inflation_data,
    load_comprehensive_rates_data,
    load_comprehensive_employment_data,
    download_fama_french_factors,
)

# Load from cache (or download if not cached)
inflation = load_comprehensive_inflation_data()

# Force re-download (ignores cache)
rates = load_comprehensive_rates_data(force_refresh=True)

# Custom date range
employment = load_comprehensive_employment_data(
    start_date='2010-01-01',
    end_date='2023-12-31'
)
```

### Run as script

```bash
uv run python clean.py
```

## Data Sources

| Category | Source | Functions |
|----------|--------|-----------|
| Stock prices | Yahoo Finance | `get_stock_data()` |
| VIX | FRED | `download_vix_data()` |
| Fama-French factors | Kenneth French Library | `download_fama_french_factors()`, `download_industry_portfolios()` |
| SEC Form 4 filings | SEC EDGAR | `Form4Downloader` |
| Inflation | FRED (CPI, PCE, PPI, etc.) | `load_comprehensive_inflation_data()` |
| Interest rates | FRED (Treasuries, SOFR, spreads) | `load_comprehensive_rates_data()` |
| Industrial production | FRED | `load_comprehensive_ip_data()` |
| Money supply (M2) | FRED (M1, M2, Fed balance sheet) | `load_comprehensive_m2_data()` |
| GDP | FRED (headline, components, by industry) | `load_comprehensive_gdp_data()` |
| Employment | FRED (payrolls, JOLTS, wages) | `load_comprehensive_employment_data()` |
| Additional macro | FRED (sentiment, housing, dollar) | `load_additional_macro_data()` |
| News sentiment | Guardian, NYT, Reddit | `load_news_data()` |

## Caching

Downloaded data is cached as `.parquet` files in `./data/fred/` and `./fama_french_data/`. Subsequent calls load from cache unless:

- The cache file doesn't exist
- `force_refresh=True` is passed (available on comprehensive loader functions)
- A different date range is requested (cache keys include date range)

## Project Structure

```
signalsandsystems/
  clean.py                                  # Main data pipeline (all loaders, cleaners, aggregators)
  main.py                                   # Analysis entry point
  Culture_War_Companies_160_fullmeta.csv    # Input: culture war companies metadata
  .env                                      # API keys (not tracked)
  .env.example                              # API key template
  data/fred/                                # Cached FRED data (not tracked)
  fama_french_data/                         # Cached Fama-French data (not tracked)
  news_data/                                # Cached news data (not tracked)
  sec_form4_data/                           # Cached SEC filings (not tracked)
```
