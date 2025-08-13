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

        days_to_process = 1000
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

        # Actual buy/sell signals from labels
        buy_indices = np.where(tr_labels == 2)[0]
        sell_indices = np.where(tr_labels == 1)[0]

        # Plot actual buy/sell
        plt.figure(figsize=(14, 6))
        plt.plot(aligned_prices, label='Price')
        plt.plot(aligned_prices.index[buy_indices], aligned_prices.iloc[buy_indices], 'g^', markersize=10, label='Buy')
        plt.plot(aligned_prices.index[sell_indices], aligned_prices.iloc[sell_indices], 'rv', markersize=10, label='Sell')
        plt.title(f"{ticker} — Filtered Buy/Sell Points with Gain Threshold & Holding Period")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- Predict all at once (batch mode) ---
        pred_test = model.predict(tr_data)
        assert pred_test.shape[0] == len(aligned_prices), "Prediction count mismatch with prices"
        
        # --- Thresholds ---
        buy_threshold = 0.9
        buy_margin_threshold = 0.2
        sell_threshold = 0.8
        sell_margin_threshold = 0.2
        
        cluster_min_count = 3
        cluster_window_days = 3
        cluster_conf_ratio = 0.90
        
        # --- Confidence margins ---
        top2_sorted = np.sort(pred_test, axis=1)[:, -2:]
        margins = top2_sorted[:, 1] - top2_sorted[:, 0]
        max_probs = np.max(pred_test, axis=1)
        pred_classes = np.argmax(pred_test, axis=1)
        
        buy_mask = (pred_classes == 2) & (max_probs > buy_threshold) & (margins > buy_margin_threshold)
        sell_mask = (pred_classes == 1) & (max_probs > sell_threshold) & (margins > sell_margin_threshold)
        
        # --- Build DataFrame for filtering ---
        df_preds = pd.DataFrame({
            "date": aligned_prices.index,
            "price": aligned_prices.squeeze().values,
            "cls": pred_classes,
            "confidence": max_probs
        })
        
        # --- Apply mask for confident predictions ---
        df_confident = df_preds[buy_mask | sell_mask].copy()
        
        # --- Cluster filter ---
        def filter_by_cluster_rule(df, min_count=3, window_days=5, conf_ratio=0.95):
            keep_indices = []
            for i, row in df.iterrows():
                cls = row["cls"]
                date = row["date"]
        
                start_date = date - pd.Timedelta(days=window_days)
                window = df[(df["cls"] == cls) &
                            (df["date"] >= start_date) &
                            (df["date"] <= date)]
        
                if not window.empty:
                    max_conf = window["confidence"].max()
                    high_conf_count = (window["confidence"] >= max_conf * conf_ratio).sum()
                    if high_conf_count >= min_count:
                        keep_indices.append(i)
        
            return df.loc[keep_indices]
        
        df_clustered = filter_by_cluster_rule(
            df_confident,
            min_count=cluster_min_count,
            window_days=cluster_window_days,
            conf_ratio=cluster_conf_ratio
        )
        
        # --- Separate final buy/sell indices ---
        buy_pred_idxs = df_clustered.index[df_clustered["cls"] == 2]
        sell_pred_idxs = df_clustered.index[df_clustered["cls"] == 1]
        
        buy_probs = df_clustered[df_clustered["cls"] == 2]["confidence"].values
        sell_probs = df_clustered[df_clustered["cls"] == 1]["confidence"].values
        
        # --- Plot predictions ---
        plt.figure(figsize=(14, 6))
        plt.plot(aligned_prices, label='Price')
        
        # Buy predictions
        plt.plot(aligned_prices.index[buy_pred_idxs], aligned_prices.iloc[buy_pred_idxs],
                 'bo', markersize=8, label='Predicted Buy', fillstyle='none')
        for idx, prob in zip(buy_pred_idxs, buy_probs):
            x = aligned_prices.index[idx]
            y = aligned_prices.iloc[idx]
            y_text = y + 0.03 * aligned_prices.max()
            plt.plot([x, x], [y, y_text], 'b--', linewidth=0.5)
            plt.text(x, y_text, f"{prob:.2f}", color='blue', fontsize=8, ha='center')
        
        # Sell predictions
        plt.plot(aligned_prices.index[sell_pred_idxs], aligned_prices.iloc[sell_pred_idxs],
                 'ko', markersize=8, label='Predicted Sell', fillstyle='none')
        for idx, prob in zip(sell_pred_idxs, sell_probs):
            x = aligned_prices.index[idx]
            y = aligned_prices.iloc[idx]
            y_text = y - 0.03 * aligned_prices.max()
            plt.plot([x, x], [y, y_text], 'k--', linewidth=0.5)
            plt.text(x, y_text, f"{prob:.2f}", color='black', fontsize=8, ha='center')
        
        # Actual labels (for reference)
        plt.plot(aligned_prices.index[buy_indices], aligned_prices.iloc[buy_indices],
                 'g^', markersize=10, label='Buy')
        plt.plot(aligned_prices.index[sell_indices], aligned_prices.iloc[sell_indices],
                 'rv', markersize=10, label='Sell')
        
        plt.title(f"{ticker} — Cluster-Filtered Buy/Sell Predictions vs Actual")
        plt.legend()
        plt.tight_layout()
        plt.show()
