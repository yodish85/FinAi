#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 07:30:50 2025

@author: Michele
"""

import os
import importlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

import extract_features_with_fft
importlib.reload(extract_features_with_fft)

import training_model
importlib.reload(training_model)

import daily_check
importlib.reload(daily_check)

from StockFetcher import StockFetcher
from daily_check import load_model
from training_model import get_symbols_from_folder
import pandas as pd

# =========================================================
# Debug plotting of signals
# =========================================================
def plot_debug_signals(prices, raw_signals, candidate_signals, finals):
    plt.figure(figsize=(14,6))
    plt.plot(prices.index, prices.values, label="Price", color="blue")

    # 1. Raw transitions
    for idx in raw_signals:
        plt.axvline(prices.index[idx], color="red", linestyle="--", alpha=0.3)
        plt.scatter(prices.index[idx], prices.iloc[idx],
                    marker="x", color="black", s=80,
                    label="Transition" if idx == raw_signals[0] else "")
    # 2. Candidates
    for idx, sig in candidate_signals:
        if sig == "BUY":
            plt.scatter(prices.index[idx], prices.iloc[idx], marker="^", color="gold", s=120, label="BUY candidate" if idx == candidate_signals[0][0] else "")
        else:
            plt.scatter(prices.index[idx], prices.iloc[idx], marker="v", color="gold", s=120, label="SELL candidate" if idx == candidate_signals[0][0] else "")

    # 3. Finals → support both dataframe or list
    if isinstance(finals, pd.DataFrame):
        if not finals.empty:
            buys = finals[finals.signal == "BUY"].entry_index
            sells = finals[finals.signal == "SELL"].entry_index
            for idx in buys:
                plt.scatter(prices.index[idx], prices.iloc[idx], marker="^", color="green", s=120, label="BUY final" if idx == buys.iloc[0] else "")
            for idx in sells:
                plt.scatter(prices.index[idx], prices.iloc[idx], marker="v", color="red", s=120, label="SELL final" if idx == sells.iloc[0] else "")
    else:
        # finals is a list of (index, signal)
        for idx, sig in finals:
            if sig == "BUY":
                plt.scatter(prices.index[idx], prices.iloc[idx], marker="^", color="green", s=120, label="BUY final" if sig == "BUY" else "")
            else:
                plt.scatter(prices.index[idx], prices.iloc[idx], marker="v", color="red", s=120, label="SELL final" if sig == "SELL" else "")

    plt.title("Signal Pipeline Debugging")
    plt.legend()
    plt.show()



# =========================================================
# Helper functions
# =========================================================
def _rolling_mode(a, w):
    s = pd.Series(a)
    out = s.rolling(w).apply(lambda x: pd.Series(x).mode().iloc[0], raw=False)
    return out.bfill().astype(int).to_numpy()

def _ema(series, span):
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

def _atr(high, low, close, period=14):
    # If you only have Close, pass it 3x (works as “vol off”)
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().to_numpy()

# =========================================================
# Entry point extraction
# =========================================================
import pandas as pd
import numpy as np

def extract_entry_points_v2(pred_probs, prices, momentum_window=10, momentum_thresh=0.05, atr_window=14, debug=False):
    import pandas as pd
    import numpy as np

    # If prices is a DataFrame, take 'Close' column
    if isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            prices = prices["Close"]
        else:
            # fall back to first column
            prices = prices.iloc[:, 0]

    raw_signals = []
    candidate_signals = []
    final_signals = []

    pred_classes = np.argmax(pred_probs, axis=1)

    # Step 1: detect transitions
    for i in range(1, len(pred_classes)):
        if pred_classes[i] != pred_classes[i-1]:
            raw_signals.append(i)

    # Step 2: Momentum
    momentum = prices.pct_change(momentum_window)

    # Step 3: ATR / Volatility
    high = prices.rolling(2).max()
    low = prices.rolling(2).min()
    close = prices
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr = tr.rolling(atr_window).mean()

    # Step 4: Filter transitions
    for idx in raw_signals:
        if idx < momentum_window or idx >= len(prices):
            continue

        mom = float(momentum.iloc[idx])  # force scalar
        vol = float(atr.iloc[idx])       # force scalar

        # Momentum logic
        if mom > momentum_thresh:
            candidate_signals.append((idx, "SELL"))
        elif mom < -momentum_thresh:
            candidate_signals.append((idx, "BUY"))
        else:
            continue

        # Volatility filter
        avg_vol = float(atr.rolling(atr_window).mean().iloc[idx])
        if vol < 0.5 * avg_vol:  # skip low volatility
            continue

        final_signals.append({
            "entry_index": idx,
            "signal": candidate_signals[-1][1]
        })

    final_df = pd.DataFrame(final_signals)

    if debug:
        print(f"Raw={len(raw_signals)}, Candidates={len(candidate_signals)}, Finals={len(final_df)}")

    return raw_signals, candidate_signals, final_df


# =========================================================
# Plot class predictions vs prices
# =========================================================
def plot_classes_vs_price(aligned_prices, pred_classes, df_entries):
    plt.figure(figsize=(14,6))
    plt.plot(aligned_prices.index, aligned_prices.values, label="Price", color="blue")

    # Plot predicted classes overlay
    plt.scatter(aligned_prices.index, aligned_prices.values, 
                c=pred_classes, cmap="coolwarm", alpha=0.3, label="Predicted class")

    # Plot entries (support df or list)
    if isinstance(df_entries, pd.DataFrame):
        if not df_entries.empty:
            buys = df_entries[df_entries.signal == "BUY"].entry_index
            sells = df_entries[df_entries.signal == "SELL"].entry_index
            for idx in buys:
                plt.scatter(aligned_prices.index[idx], aligned_prices.iloc[idx], marker="^", color="green", s=120, label="BUY entry")
            for idx in sells:
                plt.scatter(aligned_prices.index[idx], aligned_prices.iloc[idx], marker="v", color="red", s=120, label="SELL entry")
    else:
        # list of (index, signal)
        for idx, sig in df_entries:
            if sig == "BUY":
                plt.scatter(aligned_prices.index[idx], aligned_prices.iloc[idx], marker="^", color="green", s=120, label="BUY entry")
            elif sig == "SELL":
                plt.scatter(aligned_prices.index[idx], aligned_prices.iloc[idx], marker="v", color="red", s=120, label="SELL entry")

    plt.title("Predicted Classes vs Price")
    plt.legend()
    plt.show()



# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    data_path = "/Users/admin/FinAi/market_data/train"
    tickers = get_symbols_from_folder(data_path)
    
    # Load model
    model_path = "/Users/admin/FinAi"
    model = load_model(model_path)

    for ticker in tickers:
        print(f"\n--- Training model for {ticker} ---\n")

        # Fetch latest data
        print("🔄 Fetching fresh data...")
        fetcher = StockFetcher(base_path=data_path)
        fetcher.fetch_and_save(ticker, data_path)

        days_to_process = 2000
        doBalance = False

        result = extract_features_with_fft.extract_features_with_fft(
            [ticker], data_path, True, 'daily', days_to_process, doBalance
        )

        if result is None:
            print(f"[Warning] Skipping ticker {ticker} — feature extraction failed or not enough data.")
            continue

        tr_data, tr_labels, tr_symbols = result
        tr_labels = np.array(tr_labels)

        # Load price history
        df = yf.download(ticker, start='2015-01-01')
        if df.empty:
            print(f"[Warning] No price data for {ticker}")
            continue

        # Align prices with tr_labels (fixes misalignment)
        aligned_prices = df["Close"].iloc[-len(tr_labels):]
        if len(aligned_prices) != len(tr_labels):
            print(f"[Error] Label-price mismatch for {ticker}")
            continue

        # --- Predict all at once (batch mode) ---
        pred_test = model.predict(tr_data)
        assert pred_test.shape[0] == len(aligned_prices), "Prediction count mismatch with prices"

        # Extract signals
        raw, candidates, finals = extract_entry_points_v2(pred_test, aligned_prices, debug=True)

        # Plot debug pipeline
        plot_debug_signals(aligned_prices, raw, candidates, finals)

        # Also plot class overlays
        pred_classes = np.argmax(pred_test, axis=1)
        plot_classes_vs_price(aligned_prices, pred_classes, finals)
