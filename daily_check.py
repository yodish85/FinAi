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
import model_validation
importlib.reload(model_validation)

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
    days_to_process = 270 # need at least 200 days to compute the moving avg + 60 to compute the last day's prediction
    doBalance = False

    # Get all symbols
    directory = '/Users/admin/FinAi/market_data/'
    files = os.listdir(directory)
    # Get tickers from training
    all_symbols = training_model.get_symbols_from_folder(directory)
        
    # --- Initialize before the loop ---
    sell_probs_list = []
    sell_tickers_list = []
    buy_probs_list = []
    buy_tickers_list = []

    # Load model
    model = load_model('/Users/admin/FinAi')
    all_symbols = ["PRI", "TTD", "ALEX", "SNV", "MOS"]

    for i, ticker in enumerate(all_symbols, start=1):
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
        
        # Buys
        buy_raw = pred_classes == 1
        
        # require confidence >= 0.99
        confidence_mask = confidences >= 0.99995
        
        # strict buy mask
        buy_strict = buy_raw & confidence_mask
        close_prices = df["Close"]
        last_1000 = close_prices.tail(741).to_numpy().ravel() 
        minima_mask = model_validation.strict_rolling_extrema(last_1000, lookback=5, mode="min")
        trend_mask = model_validation.ma_trend_filter(last_1000, short=5, long=20, mode="bull")
        
        score = (
            buy_strict.astype(int)[-1] +
            minima_mask.astype(int)[-1] +
            trend_mask.astype(int)[-1]
        )
        
        # Require at least 2 out of 3 conditions
        buy_mask = score >= 2        

        # Sells
        sell_raw = pred_classes == 0
        
        # require confidence >= 0.99
        confidence_mask = confidences >= 0.9999
        
        # strict buy mask
        sell_strict = sell_raw & confidence_mask
        
        maxima_mask = model_validation.strict_rolling_extrema(last_1000, lookback=5, mode="max")
        trend_mask = model_validation.ma_trend_filter(last_1000, short=5, long=20, mode="bear")
        
        score = (
            sell_strict.astype(int)[-1] +
            maxima_mask.astype(int)[-1] +
            trend_mask.astype(int)[-1]
        )
        
        # Require at least 2 out of 3 conditions
        sell_mask = score >= 2        
                
        # --- Separate final buy/sell indices ---
        buy_pred_idxs = np.where(buy_mask)[0]
        sell_pred_idxs = np.where(sell_mask)[0]

        if sell_pred_idxs.size > 0:
            best_idx = sell_pred_idxs
            sell_probs_list.append(confidences)
            sell_tickers_list.append(ticker)
        
        if buy_pred_idxs.size > 0:
            best_idx = buy_pred_idxs
            buy_probs_list.append(confidences)
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
            