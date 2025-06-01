#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:00:44 2025

@author: Michele
"""

import yfinance as yf
import os
from datetime import datetime

import pandas as pd

# Download the list of DJIA companies
djia_tickers = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

print(f"Found {len(djia_tickers)} tickers.")

# Now you can run the same downloading loop as before
save_folder = '/Users/admin/Desktop/financial_ai_model/djia'
start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

os.makedirs(save_folder, exist_ok=True)

for ticker in djia_tickers:
    print(f"Downloading {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        file_path = os.path.join(save_folder, f"{ticker}.csv")
        data.to_csv(file_path)
        print(f"Saved {ticker} data to {file_path}")
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

print("All done!")

