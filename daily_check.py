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

def plot_ticker_probs(tickers, probs, close_prices=None, title="", color="blue", side="buy", savepath=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    tickers = np.array(tickers)
    probs = np.array([
        float(p[0]) if isinstance(p, (list, np.ndarray)) else 
        float(p.iloc[0]) if isinstance(p, pd.Series) else 
        float(p)
        for p in probs
    ])

    if close_prices is not None:
        close_prices = np.array([float(c) for c in close_prices])

    if len(tickers) != len(probs):
        raise ValueError(f"Length mismatch: {len(tickers)} tickers vs {len(probs)} probs")
    if close_prices is not None and len(close_prices) != len(tickers):
        raise ValueError(f"Length mismatch: {len(tickers)} tickers vs {len(close_prices)} close_prices")

    # Sort ascending by probability
    sorted_idx = np.argsort(probs)
    sorted_tickers = tickers[sorted_idx]
    sorted_probs = probs[sorted_idx]
    if close_prices is not None:
        sorted_closes = close_prices[sorted_idx]
    else:
        sorted_closes = None

    # Compute limit prices
    if sorted_closes is not None:
        if side.lower() == "buy":
            limit_prices = sorted_closes * 1.005  # +0.5% for buy
        elif side.lower() == "sell":
            limit_prices = sorted_closes * 0.995  # -0.5% for sell
        else:
            raise ValueError("side must be 'buy' or 'sell'")
    else:
        limit_prices = None

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(tickers) * 0.35), 6))
    bars = ax.bar(range(len(sorted_tickers)), sorted_probs, color=color)

    ax.set_xticks(range(len(sorted_tickers)))
    ax.set_xticklabels(sorted_tickers, rotation=90)
    ax.set_title(title)
    ax.set_ylabel("Probability")

    # Dynamic y-limit (space for labels)
    y_max = max(sorted_probs)
    ax.set_ylim(0, y_max * 1.4)

    # Add labels above bars (non-overlapping)
    # Dynamic y-limit (large headroom)
    # Expand y-limit to allow two label rows
    y_max = max(sorted_probs)
    ax.set_ylim(0, y_max * 1.6)
    
    for i, bar in enumerate(bars):
        prob = sorted_probs[i]
    
        if sorted_closes is not None:
            close = sorted_closes[i]
            limit = limit_prices[i]
            label = f"{prob:.2f}\n${close:.2f}\n${limit:.2f}"
        else:
            label = f"{prob:.2f}"
    
        # Alternate label height (THIS is the key)
        y_offset = 0.03 if i % 2 == 0 else 0.18
    
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False
        )


    fig.tight_layout()
    plt.show() 
    if savepath: 
        fig.savefig(savepath, bbox_inches="tight", dpi=200) 
        plt.close(fig)


     
if __name__ == "__main__":
    
    data_path = "/Users/admin/FinAi/market_data"
    days_to_process = 300 # need at least 200 days to compute the moving avg + 60 to compute the last day's prediction
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
    sell_close_prices_list = []
    buy_close_prices_list = []
    
    # Load model
    model = load_model('/Users/admin/FinAi/')
    #all_symbols = ["STX", "TTD", "ALEX", "SNV", "MOS"]
    
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
        
        prices_np = aligned_prices.to_numpy().ravel()   # ensures 1D

        confidences = np.max(pred_test, axis=1)
        buy_conf = pred_test[:, 2]
        sell_conf = pred_test[:, 1]

        pred_classes = np.argmax(pred_test, axis=1)
        
        conf_th = 0.5
        
        # Basic: use raw confidences, 3-day rising window, default classes (buy_class=2,sell_class=1)
        res = comprehensive_validation.directional_confidence_signals(
            pred_test,
            trend_window=3,
            conf_th=conf_th,
        )
        
        # Apply price filters
        buy_mask = res['buy_mask'].copy()
        sell_mask = res['sell_mask'].copy()
        
        # Ensure last_close is float
        last_close = float(aligned_prices.iloc[-1])
        
        # Only consider signals where mask is True AND confidence >= conf_th
        buy_pred_idxs = np.where(buy_mask & (buy_conf >= conf_th))[0]
        sell_pred_idxs = np.where(sell_mask & (sell_conf >= conf_th))[0]
        
        # Append the last valid sell signal, if any
        if sell_pred_idxs.size > 0:
            last_sell_idx = sell_pred_idxs[-1]
            sell_probs_list.append(sell_conf[last_sell_idx])
            sell_tickers_list.append(ticker)
            sell_close_prices_list.append(last_close)  # Close price reference
        
        # Append the last valid buy signal, if any
        if buy_pred_idxs.size > 0:
            last_buy_idx = buy_pred_idxs[-1]
            buy_probs_list.append(buy_conf[last_buy_idx])
            buy_tickers_list.append(ticker)
            buy_close_prices_list.append(last_close)   # Close price reference


    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Limit to first 30 tickers if the list is longer
    max_tickers = 30
    
    # Plot Buy predictions
    plot_ticker_probs(
        tickers=buy_tickers_list[:max_tickers],
        probs=buy_probs_list[:max_tickers],
        close_prices=buy_close_prices_list[:max_tickers],
        title="Buy Predictions",
        color="green",
        side="buy",
        savepath=f"predictions/buy_predictions_{timestamp}.png"
    )

    # Example: assuming aligned_prices contains latest close for each ticker
    plot_ticker_probs(
        tickers=sell_tickers_list[:max_tickers],
        probs=sell_probs_list[:max_tickers],
        close_prices=sell_close_prices_list[:max_tickers],
        title="Sell Predictions",
        color="red",
        side="sell",
        savepath=f"predictions/sell_predictions_{timestamp}.png"
    )   