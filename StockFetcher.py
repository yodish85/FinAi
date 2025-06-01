#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 21:14:45 2025

@author: Michele
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to fetch market data and split into train vs validation:
 - train: sp500, nasdaq100, cryptos
 - validation: dow_jones, russell2000, ftse100, dax, nikkei225
"""

import os
import re
from datetime import datetime

import requests
import yfinance as yf
from bs4 import BeautifulSoup


class StockFetcher:
    def __init__(self, base_path, start="2010-01-01", end=None):
        self.start = start
        self.end = end or datetime.today().strftime("%Y-%m-%d")
        self.base = base_path

        self.train_indices = ["sp500", "nasdaq100"]
        self.val_indices = ["dow_jones", "russell2000", "ftse100", "dax", "nikkei225"]

        self.wiki_urls = {
            "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
            "nasdaq100": "https://en.wikipedia.org/wiki/NASDAQ-100",
            "dow_jones": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
            "russell2000": "https://en.wikipedia.org/wiki/Russell_2000_Index",
            "ftse100": "https://en.wikipedia.org/wiki/FTSE_100",
            "dax": "https://en.wikipedia.org/wiki/DAX",
            "nikkei225": "https://en.wikipedia.org/wiki/Nikkei_225",
        }

        self.fallback = {
            "sp500": ["SPY"],
            "nasdaq100": ["QQQ"],
            "dow_jones": [
                'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
                'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
                'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
            ],
            "russell2000": ["IWM"],
            "ftse100": ["VUKE.L"],
            "dax": ["DAXEX.DE"],
            "nikkei225": ["EWJ"],
        }

        self.cryptos = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "BNB-USD", "TRX-USD", "DOGE-USD"]

    def scrape_index_tickers(self, idx_name):
        """Scrape the Wikipedia table for idx_name, fall back if anything fails."""
        url = self.wiki_urls[idx_name]
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html5lib")
            tables = soup.find_all("table", {"class": "wikitable"})
            for tbl in tables:
                headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
                for key in ("symbol", "ticker", "epic"):
                    if any(key in h for h in headers):
                        col = next(i for i, h in enumerate(headers) if key in h)
                        tickers = []
                        for row in tbl.find_all("tr")[1:]:
                            cells = row.find_all(["td", "th"])
                            if len(cells) > col:
                                txt = cells[col].get_text(strip=True)
                                txt = re.sub(r"\[.*?\]", "", txt)  # remove footnote refs
                                if txt:
                                    tickers.append(txt)
                        print(f" • {idx_name}: scraped {len(tickers)} symbols")
                        return tickers
            raise ValueError("No ticker column found")
        except Exception as e:
            print(f"⚠️ {idx_name} scrape failed ({e}), using fallback ETF")
            return self.fallback[idx_name]

    def fetch_and_save(self, symbols, out_folder):
        """Download and save each symbol's CSV into out_folder."""
        os.makedirs(out_folder, exist_ok=True)
        for sym in symbols:
            try:
                df = yf.download(sym, start=self.start, end=self.end, progress=False)
                if df.empty:
                    print(f"   ⚠️ {sym}: no data")
                    continue
                path = os.path.join(out_folder, f"{sym}.csv")
                df.to_csv(path)
            except Exception as ex:
                print(f"   ❌ {sym}: {ex}")

    def run(self):
        """Main pipeline to fetch and store train/validation data"""
        # TRAIN: scrape + crypto
        train_syms = []
        for idx in self.train_indices:
            train_syms += self.scrape_index_tickers(idx)
        train_syms += self.cryptos

        print(f"\n=== TRAIN set: {len(train_syms)} symbols ===")
        self.fetch_and_save(train_syms, os.path.join(self.base, "train"))

        # VALIDATION: scrape only
        val_syms = []
        for idx in self.val_indices:
            val_syms += self.scrape_index_tickers(idx)

        print(f"\n=== VALIDATION set: {len(val_syms)} symbols ===")
        self.fetch_and_save(val_syms, os.path.join(self.base, "validation"))

        print("\n✅ All downloads complete.")

