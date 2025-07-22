#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 07:30:50 2025

@author: Michele
"""

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
import yfinance as yf

import daily_check
import training_model


from daily_check import load_model
from training_model import get_symbols_from_folder

if __name__ == "__main__":
    
    directory = '/Users/admin/FinAi/market_data/train'
    files = os.listdir(directory)
    # Get tickers from training
    tickers = training_model.get_symbols_from_folder(directory)
    for ticker in tickers:
        print(f"\n--- Training model for {ticker} ---\n")
    
        # ----------------------------
        # Fresh Fetch & Feature Generation (Fixed)
        # ----------------------------
        print("🔄 Fetching fresh data...")
        
        # Fetch last 180 calendar days
        data_path = "/Users/admin/FinAi/market_data/train"
        fetcher = StockFetcher(base_path=data_path)
        fetcher.fetch_and_save(ticker, data_path)
    
        days_to_process = 1000
        doBalance = False
        directory = []
        
        result = extract_features_with_fft.extract_features_with_fft([ticker], data_path, True, 'daily', days_to_process, doBalance)

        if result is None:
            print(f"[Warning] Skipping ticker {ticker} — feature extraction failed or not enough data.")
            continue  # or `return`, depending on your program structure
        
        tr_data, tr_labels, tr_symbols = result

        df = yf.download(ticker, start='2015-01-01')
    
        # retain the last N days
        days_to_retain = len(tr_labels)
        prices = df["Close"].tail(days_to_retain)
        
    
        # Plotting
        plt.figure(figsize=(14, 6))
        plt.plot(prices, label='Price')
        
        # Convert tr_labels to a NumPy array if it isn't already
        tr_labels = np.array(tr_labels)
        
        # Get indices where tr_labels == 2 (Buy)
        buy_indices = np.where(tr_labels == 2)[0]
        # Get indices where tr_labels == 1 (Sell)
        sell_indices = np.where(tr_labels == 1)[0]
        
        # Plot buy signals
        plt.plot(prices.index[buy_indices], prices.iloc[buy_indices], 'g^', markersize=10, label='Buy')
        
        # Plot sell signals
        plt.plot(prices.index[sell_indices], prices.iloc[sell_indices], 'rv', markersize=10, label='Sell')
        
        plt.title("Filtered Buy/Sell Points with Gain Threshold & Holding Period")
        plt.legend()
        plt.show()
      
        # 3. Load latest model
        importlib.reload(daily_check)  # Reload the module, not the function
        model_path = "/Users/admin/FinAi"
        model = daily_check.load_model(model_path)
        pred_test = np.zeros((days_to_retain, 3))  # pre-allocate for 3 outputs per sample
    
        # Compute predictions
        for i in range(tr_data.shape[0]):
            window_data = tr_data[i]  # sample has shape (30, 76)
            window_data = np.expand_dims(window_data, axis=0)  # shape: (1, 30, 76)
            pred_test[i,:] = model.predict(window_data)
        
        # --- Settings ---
        buy_threshold = 0.73
        buy_margin_threshold = 0.2
        
        sell_threshold = 0.69
        sell_margin_threshold = 0.2
        
        # --- Compute top-2 margins ---
        top2_sorted = np.sort(pred_test, axis=1)[:, -2:]
        margins = top2_sorted[:, 1] - top2_sorted[:, 0]
        
        # --- Get max prediction values and predicted classes ---
        max_probs = np.max(pred_test, axis=1)
        pred_classes = np.argmax(pred_test, axis=1)
        
        # --- Get confident BUY predictions ---
        buy_mask = (pred_classes == 2) & \
                   (max_probs > buy_threshold) & \
                   (margins > buy_margin_threshold)
        buy_pred_idxs = np.where(buy_mask)[0]
        buy_probs = pred_test[buy_pred_idxs, 2]  # Probabilities for class 2 (Buy)
        
        # --- Get confident SELL predictions ---
        sell_mask = (pred_classes == 1) & \
                    (max_probs > sell_threshold) & \
                    (margins > sell_margin_threshold)
        sell_pred_idxs = np.where(sell_mask)[0]
        sell_probs = pred_test[sell_pred_idxs, 1]  # Probabilities for class 1 (Sell)
        
        # --- Plot price curve ---
        plt.figure(figsize=(14, 6))
        plt.plot(prices, label='Price')
        
        # --- Plot predicted Buys ---
        plt.plot(prices.index[buy_pred_idxs], prices.iloc[buy_pred_idxs], 
                 'bo', markersize=8, label='Predicted Buy', fillstyle='none')
        for idx, prob in zip(buy_pred_idxs, buy_probs):
            x = prices.index[idx]
            y = prices.iloc[idx]
            y_text = y + 0.03 * prices.max()  # Raise text slightly above point
            plt.plot([x, x], [y, y_text], 'b--', linewidth=0.5)  # Connecting line
            plt.text(x, y_text, f"{prob:.2f}", color='blue', fontsize=8, ha='center')
        
        # --- Plot predicted Sells ---
        plt.plot(prices.index[sell_pred_idxs], prices.iloc[sell_pred_idxs], 
                 'ko', markersize=8, label='Predicted Sell', fillstyle='none')
        for idx, prob in zip(sell_pred_idxs, sell_probs):
            x = prices.index[idx]
            y = prices.iloc[idx]
            y_text = y - 0.03 * prices.max()  # Lower text below point
            plt.plot([x, x], [y, y_text], 'k--', linewidth=0.5)  # Connecting line
            plt.text(x, y_text, f"{prob:.2f}", color='black', fontsize=8, ha='center')
        
        # --- Plot actual buy/sell signals (optional) ---
        plt.plot(prices.index[buy_indices], prices.iloc[buy_indices], 
                 'g^', markersize=10, label='Buy')
        plt.plot(prices.index[sell_indices], prices.iloc[sell_indices], 
                 'rv', markersize=10, label='Sell')
        
        # Final touches
        plt.title("Buy/Sell Predictions vs Actual")
        plt.legend()
        plt.tight_layout()
        plt.show()
