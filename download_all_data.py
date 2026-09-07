import sys
import os
import datetime

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import yfinance as yf

COMPANIES = {
    "SBIN": {"name": "State Bank of India", "ticker": "SBIN.NS", "file": "sbin_historical_2025_2026.csv"},
    "RELIANCE": {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "file": "reliance_historical_2025_2026.csv"},
    "TCS": {"name": "Tata Consultancy Services", "ticker": "TCS.NS", "file": "tcs_historical_2025_2026.csv"},
    "INFY": {"name": "Infosys", "ticker": "INFY.NS", "file": "infy_historical_2025_2026.csv"},
    "HDFCBANK": {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "file": "hdfcbank_historical_2025_2026.csv"},
    "ICICIBANK": {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "file": "icicibank_historical_2025_2026.csv"},
    "ITC": {"name": "ITC", "ticker": "ITC.NS", "file": "itc_historical_2025_2026.csv"},
    "LT": {"name": "Larsen & Toubro", "ticker": "LT.NS", "file": "lt_historical_2025_2026.csv"},
    "BHARTIARTL": {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS", "file": "bhartiartl_historical_2025_2026.csv"},
    "TATAMOTORS": {"name": "Tata Motors", "ticker": "TATAMOTORS.NS", "file": "tatamotors_historical_2025_2026.csv"}
}

print("Fetching real NSE historical data for all 10 companies (1 April 2025 - 31 March 2026)...")

for code, info in COMPANIES.items():
    csv_file = info["file"]
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f"[{code}] {info['name']}: Cached ({len(df)} records in {csv_file})")
        continue

    print(f"[{code}] Downloading {info['name']} ({info['ticker']})...")
    raw = yf.download(info['ticker'], start='2025-04-01', end='2026-04-01')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.sort_values('Date').drop_duplicates(subset=['Date']).dropna(subset=['Close'])
    df = df[df['Date'] <= datetime.date(2026, 3, 31)].reset_index(drop=True)
    df.to_csv(csv_file, index=False)
    print(f"[{code}] Saved {len(df)} records to {csv_file}")

print("All 10 companies data ready!")
