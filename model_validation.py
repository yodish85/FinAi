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
    tickers = tickers[0:100]
    for ticker in tickers:
        print(f"\n--- Training model for {ticker} ---\n")
    
        # ----------------------------
        # Fresh Fetch & Feature Generation (Fixed)
        # ----------------------------
        print("🔄 Fetching fresh data...")
        
        # Fetch last 180 calendar days
        data_path = "/Users/admin/FinAi/model_validation"
        fetcher = StockFetcher(base_path=data_path)
        fetcher.fetch_and_save(ticker, data_path)
    
        days_to_process = 1000
        doBalance = False
        directory = []
        
        tr_data, tr_labels, tr_symbols =  \
            extract_features_with_fft.extract_features_with_fft(ticker, data_path, True, 'daily', days_to_process, doBalance)
        
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
        threshold = 0.85
        margin_threshold = 0.4
        
        # --- Compute top-2 margins ---
        top2_sorted = np.sort(pred_test, axis=1)[:, -2:]
        margins = top2_sorted[:, 1] - top2_sorted[:, 0]
        
        # --- Get confident prediction indexes ---
        confident_idxs = np.where((np.max(pred_test, axis=1) > threshold) & (margins > margin_threshold))[0]
        
        # --- Extract predicted and true classes for confident samples ---
        confident_preds = np.argmax(pred_test[confident_idxs], axis=1)
        
        # Convert to arrays just to be safe
        confident_preds = np.array(confident_preds)
        confident_idxs = np.array(confident_idxs)
        
        # Find indices where confident_preds == 1 or 2
        buy_pred_idxs = confident_idxs[confident_preds == 2]
        sell_pred_idxs = confident_idxs[confident_preds == 1]
        
        # Plot price curve
        plt.figure(figsize=(14, 6))
        plt.plot(prices, label='Price')
    
        # Predicted buy (1) → empty blue circles
        plt.plot(prices.index[buy_pred_idxs], prices.iloc[buy_pred_idxs], 
                 'bo', markersize=8, label='Predicted Buy', fillstyle='none')
        
        # Predicted sell (2) → empty black circles
        plt.plot(prices.index[sell_pred_idxs], prices.iloc[sell_pred_idxs], 
                 'ko', markersize=8, label='Predicted Sell', fillstyle='none')
        
        # Actual buy/sell signals (optional)
        plt.plot(prices.index[buy_indices], prices.iloc[buy_indices], 'g^', markersize=10, label='Buy')
        plt.plot(prices.index[sell_indices], prices.iloc[sell_indices], 'rv', markersize=10, label='Sell')
        
        # Final touches
        plt.title("Buy/Sell Predictions vs Actual")
        plt.legend()
        plt.show()
