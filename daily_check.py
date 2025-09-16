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

if __name__ == "__main__":
    
    data_path = "/Users/admin/FinAi/market_data"
    days_to_process = 263 # need at least 200 days to compute the moving avg + 60 to compute the last day's prediction + 3 to compute the clusters
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
        
        # Actual buy/sell signals from labels
        buy_indices = np.where(tr_labels == 2)[0]
        sell_indices = np.where(tr_labels == 1)[0]
        '''
        # Plot actual buy/sell
        plt.figure(figsize=(14, 6))
        plt.plot(aligned_prices, label='Price')
        plt.plot(aligned_prices.index[buy_indices], aligned_prices.iloc[buy_indices], 'g^', markersize=10, label='Buy')
        plt.plot(aligned_prices.index[sell_indices], aligned_prices.iloc[sell_indices], 'rv', markersize=10, label='Sell')
        plt.title(f"{ticker} — Filtered Buy/Sell Points with Gain Threshold & Holding Period")
        plt.legend()
        plt.tight_layout()
        plt.show()
        '''
        # --- Predict all at once (batch mode) ---
        pred_test = model.predict(tr_data)
        assert pred_test.shape[0] == len(aligned_prices), "Prediction count mismatch with prices"
        
        # --- Thresholds ---
        buy_threshold = 0.9
        buy_margin_threshold = 0.0
        sell_threshold = 0.82
        sell_margin_threshold = 0.0
        
        cluster_min_count = 1
        cluster_window_days = 1
        cluster_conf_ratio = 0.9
        
        # --- Moving Average (50-day) ---
        prices_np = aligned_prices.to_numpy().ravel()   # ensures 1D

        # 50-day simple moving average (SMA) via convolution
        # 50-day simple moving average (SMA)
        ma50 = df["Close"].rolling(window=50, min_periods=1).mean()
        n_preds = len(pred_test)
        ma50_last = ma50.iloc[-n_preds:].to_numpy()

        # Match last N prices/MA with pred_test length
        prices_np_last = prices_np[-n_preds:]
        ma50_last = ma50[-n_preds:].to_numpy()

        # --- Confidence margins ---
        top2_sorted = np.sort(pred_test, axis=1)[:, -2:]
        margins = top2_sorted[:, 1] - top2_sorted[:, 0]
        max_probs = np.max(pred_test, axis=1)
        pred_classes = np.argmax(pred_test, axis=1)
        
        # --- Buy/Sell conditions with MA filter ---
        buy_mask = (
            (pred_classes == 2) &
            (max_probs > buy_threshold) &
            (margins > buy_margin_threshold) &
            (prices_np_last < ma50_last)     
        )
        
        sell_mask = (
            (pred_classes == 1) &
            (max_probs > sell_threshold) &
            (margins > sell_margin_threshold) &
            (prices_np_last > ma50_last)     
        )
        
        # --- Build DataFrame for filtering ---
        df_preds = pd.DataFrame({
            "date": aligned_prices.index,
            "price": aligned_prices.squeeze().values,
            "cls": pred_classes,
            "confidence": max_probs
        })
        
        # --- Apply mask for confident predictions ---
        df_confident = df_preds[buy_mask | sell_mask].copy()
        
        # --- Keep only the last day ---
        last_day = aligned_prices.index[-1]
        df_confident = df_confident[df_confident["date"] == last_day]
        
        # --- Apply cluster filter (optional here, but with one day it will just pass through) ---
        df_clustered = model_validation.filter_by_cluster_rule(
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
        
        if not sell_pred_idxs.empty:
            best_idx = df_clustered.loc[sell_pred_idxs, "confidence"].idxmax()
            sell_probs_list.append(df_clustered.loc[best_idx, "confidence"])
            sell_tickers_list.append(ticker)
        
        if not buy_pred_idxs.empty:
            best_idx = df_clustered.loc[buy_pred_idxs, "confidence"].idxmax()
            buy_probs_list.append(df_clustered.loc[best_idx, "confidence"])
            buy_tickers_list.append(ticker)




    import numpy as np
    
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
         
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot Buy predictions
    plot_ticker_probs(buy_tickers_list, buy_probs_list, "Buy Predictions", color="green", savepath=f"buy_predictions_{timestamp}.png")
    
    # Plot Sell predictions
    plot_ticker_probs(sell_tickers_list, sell_probs_list, "Sell Predictions", color="red", savepath=f"sell_predictions_{timestamp}.png")
        
            