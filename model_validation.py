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
from training_model import get_symbols_from_folder
import pandas as pd

def strict_rolling_extrema(prices, lookback=5, threshold=0.01, mode="min"):
    """
    Detects strict local extrema (minima for buys, maxima for sells).

    Args:
        prices (array-like): Price series.
        lookback (int): Window size for rolling min/max and mean.
        threshold (float): Deviation from rolling mean required (e.g. 0.01 = 1%).
        mode (str): "min" for minima (buy), "max" for maxima (sell).

    Returns:
        np.ndarray: Boolean mask of extrema.
    """
    s = pd.Series(prices)
    rolling_mean = s.rolling(lookback, center=False, min_periods=1).mean()

    if mode == "min":  # Buy condition
        rolling_ext = s.rolling(lookback, center=False, min_periods=1).min()
        mask = (s == rolling_ext) & (s < (rolling_mean * (1 - threshold)))

    elif mode == "max":  # Sell condition
        rolling_ext = s.rolling(lookback, center=False, min_periods=1).max()
        mask = (s == rolling_ext) & (s > (rolling_mean * (1 + threshold)))

    else:
        raise ValueError("mode must be 'min' or 'max'")

    return mask.fillna(False).to_numpy()

def ma_trend_filter(prices, short=5, long=20, margin=0.01, mode="bull"):
    """
    Trend filter for buy/sell signals.

    Args:
        prices (array-like): Price series.
        short (int): Short moving average window.
        long (int): Long moving average window.
        margin (float): % margin above/below long MA.
        mode (str): "bull" for buy filter, "bear" for sell filter.

    Returns:
        np.ndarray: Boolean mask of trend conditions.
    """
    s = pd.Series(prices)
    ma_short = s.rolling(short, min_periods=1).mean()
    ma_long = s.rolling(long, min_periods=1).mean()

    # slopes
    ma_short_diff = ma_short.diff()
    ma_long_diff = ma_long.diff()

    if mode == "bull":  # Buy trend
        mask = (ma_short > ma_long) & (ma_short_diff > 0) & (ma_long_diff > 0)
        mask = mask & (s > ma_long * (1 + margin))

    elif mode == "bear":  # Sell trend
        mask = (ma_short < ma_long) & (ma_short_diff < 0) & (ma_long_diff < 0)
        mask = mask & (s < ma_long * (1 - margin))

    else:
        raise ValueError("mode must be 'bull' or 'bear'")

    return mask.fillna(False).to_numpy()

def decluster_signals(signal_mask, min_gap=10, mode="first"):
    """
    Decluster signals by enforcing a minimum gap between them.
    
    Args:
        signal_mask (np.ndarray): Boolean mask of signals.
        min_gap (int): Minimum number of bars between signals.
        mode (str): 
            "first" -> keep the first signal in each cluster
            "last"  -> keep the last signal in each cluster
            "max_conf" -> keep the signal with highest confidence (requires confidences)
    
    Returns:
        np.ndarray: Boolean mask with declustered signals.
    """
    idx = np.where(signal_mask)[0]
    keep = []
    last_kept = -min_gap

    for i in idx:
        if i - last_kept >= min_gap:
            if mode == "first":
                keep.append(i)
                last_kept = i
            elif mode == "last":
                # defer keeping until cluster ends
                if keep and keep[-1] == last_kept:
                    keep.pop()
                keep.append(i)
                last_kept = i
            else:
                raise ValueError("mode must be 'first' or 'last' for now")

    mask = np.zeros_like(signal_mask, dtype=bool)
    mask[keep] = True
    return mask



# --- Cluster filter ---        
def filter_by_cluster_rule(df, min_count=3, window_days=5, conf_ratio=0.95):
    keep_indices = []
    df = df.sort_values(["cls", "date"]).reset_index(drop=False)  # keep original index
    
    for cls, group in df.groupby("cls"):
        for i in range(len(group)):
            # Look back up to `window_days` rows, not just time
            window = group.iloc[max(0, i - window_days + 1): i + 1]
            
            # Check if dates are consecutive with no gaps
            consecutive = (window["date"].diff().dt.days.dropna() == 1).all()
            
            if len(window) >= min_count and consecutive:
                max_conf = window["confidence"].max()
                high_conf_count = (window["confidence"] >= max_conf * conf_ratio).sum()
                if high_conf_count >= min_count:
                    keep_indices.append(group.iloc[i]["index"])  # use original index
    
    return df.set_index("index").loc[keep_indices]

if __name__ == "__main__":
    data_path = "/Users/admin/FinAi/market_data/"
    tickers = get_symbols_from_folder(data_path)
    
    # Load model
    model_path = "/Users/admin/FinAi"
    model = daily_check.load_model(model_path)
    tickers = ["CAT", "LUMN", "RBA", "PRLB", "USPH", "CLF"]
    #tickers = [ "CORT"]
    for ticker in tickers:
        print(f"\n--- Training model for {ticker} ---\n")

        # Fetch latest data
        print("🔄 Fetching fresh data...")
        fetcher = StockFetcher(base_path=data_path)
        fetcher.fetch_and_save([ticker], data_path)

        days_to_process = 1001
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
        window_days = 60  # Must match what's used inside extract_features_with_fft
        aligned_prices = df["Close"].iloc[-len(tr_labels):]
        if len(aligned_prices) != len(tr_labels):
            print(f"[Error] Label-price mismatch for {ticker}")
            continue
        
        print(len(tr_labels), len(aligned_prices))
        print(aligned_prices.index[0], aligned_prices.index[-1])

        # --- Predict all at once (batch mode) ---
        pred_test = model.predict(tr_data)
        assert pred_test.shape[0] == len(aligned_prices), "Prediction count mismatch with prices"
        
        prices_np = aligned_prices.to_numpy().ravel()   # ensures 1D

        confidences = np.max(pred_test, axis=1)
        pred_classes = np.argmax(pred_test, axis=1)
        
        # Buys
        buy_raw = pred_classes == 1
        
        # require confidence >= 0.99
        confidence_mask = confidences >= 0.9
        
        # strict buy mask
        buy_strict = buy_raw & confidence_mask
        
        minima_mask = strict_rolling_extrema(prices_np, lookback=5, mode="min")
        trend_mask = ma_trend_filter(prices_np, short=5, long=20, mode="bull")
        
        score = (
            buy_strict.astype(int) +
            minima_mask.astype(int) +
            trend_mask.astype(int)
        )
        
        # Require at least 2 out of 3 conditions
        buy_mask = score >= 2        

        # Sells
        sell_raw = pred_classes == 0
        
        # require confidence >= 0.99
        confidence_mask = confidences >= 0.9
        
        # strict buy mask
        sell_strict = sell_raw & confidence_mask
        
        maxima_mask = strict_rolling_extrema(prices_np, lookback=5, mode="max")
        trend_mask = ma_trend_filter(prices_np, short=5, long=20, mode="bear")
        
        score = (
            sell_strict.astype(int) +
            maxima_mask.astype(int) +
            trend_mask.astype(int)
        )
        
        # Require at least 2 out of 3 conditions
        sell_mask = score >= 2        
                
        # --- Separate final buy/sell indices ---
        buy_pred_idxs = np.where(buy_mask)[0]
        sell_pred_idxs = np.where(sell_mask)[0]

        # --- Plot predictions ---
        plt.figure(figsize=(14, 6))
        plt.plot(aligned_prices, label='Price')
        
        # Buy predictions
        plt.plot(aligned_prices.index[buy_pred_idxs], aligned_prices.iloc[buy_pred_idxs],
                 'bo', markersize=8, label='Predicted Buy', fillstyle='none')
        plt.plot(aligned_prices.index[sell_pred_idxs], aligned_prices.iloc[sell_pred_idxs],
                 'ro', markersize=8, label='Predicted Sell', fillstyle='none')
        plt.show()
        

        