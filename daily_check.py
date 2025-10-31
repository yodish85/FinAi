#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:38:01 2025

@author: Michele
"""
import os
import importlib
import tensorflow as tf
import extract_features_with_fft
importlib.reload(extract_features_with_fft)
import training_model
importlib.reload(training_model)
from StockFetcher import StockFetcher
import numpy as np
from training_model import Attention  # Adjust path accordingly
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import pandas as pd
import comprehensive_validation
importlib.reload(comprehensive_validation)

def load_model(path):
    # Search for the latest .keras or .h5 model file
    model_files = [os.path.join(path, f) for f in os.listdir(path) 
                   if f.endswith(".keras") or f.endswith(".h5")]
    
    if not model_files:
        raise FileNotFoundError("No .keras or .h5 model found in path.")

    latest_model = max(model_files, key=os.path.getmtime)
    
    from training_model import WeightedCategoricalCrossentropy

    model = tf.keras.models.load_model(
        latest_model,
        custom_objects={'WeightedCategoricalCrossentropy': WeightedCategoricalCrossentropy}
    )

    print(f"Loaded model from: {latest_model}")
    return model

def plot_ticker_probs(tickers, probs, title, color, savepath=None):
    tickers = np.array(tickers)
    probs = np.array([
    float(p[0]) if isinstance(p, (list, np.ndarray)) else 
    float(p.iloc[0]) if isinstance(p, pd.Series) else 
    float(p)
    for p in probs
])

    if len(tickers) != len(probs):
        raise ValueError(f"Length mismatch: {len(tickers)} tickers vs {len(probs)} probs")

    # Sort ascending by probability
    sorted_idx = np.argsort(probs)
    sorted_tickers = tickers[sorted_idx]
    sorted_probs = probs[sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(sorted_tickers)), sorted_probs, color=color)
    ax.set_xticks(range(len(sorted_tickers)))
    ax.set_xticklabels(sorted_tickers, rotation=90)
    ax.set_title(title)
    ax.set_ylabel("Probability")

    # Add labels on top
    for bar, prob in zip(bars, sorted_probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{prob:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    plt.show()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
        plt.close(fig)
     
if __name__ == "__main__":
    
    data_path = "/Users/admin/FinAi/market_data"
    days_to_process = 1001 # need at least 200 days to compute the moving avg + 60 to compute the last day's prediction
    # use 1001; it complies with the validation and gives best results
    doBalance = False

    # Get all symbols
    directory = '/Users/admin/FinAi/market_data/'
    files = os.listdir(directory)
    # Get tickers from training
    all_symbols = training_model.get_symbols_from_folder(directory)
    
    all_symbols = comprehensive_validation.filter_sp500_tickers(all_symbols)
    
    # --- Initialize before the loop ---
    sell_probs_list = []
    sell_tickers_list = []
    buy_probs_list = []
    buy_tickers_list = []

    # Load model
    model = load_model('/Users/admin/FinAi')
    #all_symbols = ["STX", "TTD", "ALEX", "SNV", "MOS"]

    ticker_gains_map = np.load('/Users/admin/FinAi/ticker_gains_map.npy', allow_pickle=True).item()
    
    for i, ticker in enumerate(all_symbols, start=1):
        
        # Skip if ticker not in map
        if ticker not in ticker_gains_map:
            print(f"Skipping {ticker} - not in gains map")
            continue
        
        # Skip if gains are ≤20%
        if not ticker_gains_map[ticker]:
            continue
            
        print(f"Processing {i}/{len(all_symbols)}: {ticker}")
        
        # Fetch latest data
        print("🔄 Fetching fresh data...")
        fetcher = StockFetcher(base_path=data_path)
        fetcher.fetch_and_save([ticker], data_path)

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
        pred_test = model.predict(tr_data) # should have size 1
        assert pred_test.shape[0] == len(aligned_prices), "Prediction count mismatch with prices"
        
        # --- Moving Average (50-day) ---
        prices_np = aligned_prices.to_numpy().ravel()   # ensures 1D

        confidences = np.max(pred_test, axis=1)
        pred_classes = np.argmax(pred_test, axis=1)
        
        # Basic: use raw confidences, 3-day rising window, default classes (buy_class=2,sell_class=1)
        res = comprehensive_validation.directional_confidence_signals(
            pred_test,
            trend_window=3,
            conf_th=0.8,
        )
        
        # Inspect indices
        print("Buy points idx:", res['buy_idx'])
        print("Sell points idx:", res['sell_idx'])
        
        buy_mask = res["buy_mask"]
        sell_mask = res["sell_mask"]
        
        # Only populate if the LAST element is True
        buy_pred_idxs = np.where(buy_mask)[0] if buy_mask[-1] else np.array([])
        sell_pred_idxs = np.where(sell_mask)[0] if sell_mask[-1] else np.array([])
        
        # Retain only the last sell signal
        if sell_pred_idxs.size > 0:
            best_idx = sell_pred_idxs[-1]
            sell_probs_list.append(confidences[best_idx])
            sell_tickers_list.append(ticker)
        
        # Retain only the last buy signal
        if buy_pred_idxs.size > 0:
            best_idx = buy_pred_idxs[-1]
            buy_probs_list.append(confidences[best_idx])
            buy_tickers_list.append(ticker)

    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Limit to first 30 tickers if the list is longer
    max_tickers = 30
    
    # Plot Buy predictions
    plot_ticker_probs(
        buy_tickers_list[:max_tickers],
        buy_probs_list[:max_tickers],
        "Buy Predictions",
        color="green",
        savepath=f"predictions/buy_predictions_{timestamp}.png"
    )
    
    # Plot Sell predictions
    plot_ticker_probs(
        sell_tickers_list[:max_tickers],
        sell_probs_list[:max_tickers],
        "Sell Predictions",
        color="red",
        savepath=f"predictions/sell_predictions_{timestamp}.png"
    )
            