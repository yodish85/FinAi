#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 15:51:33 2025

@author: Michele
"""

import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf
import holidays
import os
import glob
import matplotlib.pyplot as plt
import pandas_ta as ta
from tqdm import tqdm  # for cleaner progress bars
import pywt
import gc
from datetime import datetime
import advanced_indicators
import importlib
importlib.reload(advanced_indicators)

# Example: US federal holidays
us_holidays = holidays.US()

def add_temporal_features(df):
    if 'Date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
        
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_week'] = df['Date'].dt.weekday          # 0 = Monday
    df['day_of_month'] = df['Date'].dt.day
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_holiday'] = df['Date'].isin(us_holidays)
    df['is_month_start'] = df['Date'].dt.is_month_start
    df['is_month_end'] = df['Date'].dt.is_month_end
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end
    return df

def get_df_list(symbol_list, base_dir):
    """
    Reads symbol CSVs that have a 3-row header (Price,..., Ticker..., Date,,,,,)
    and returns {symbol: DataFrame} with Date as datetime index and column.
    """
    dfs = {}

    for symbol in symbol_list:
        candidate = os.path.join(base_dir, f"{symbol}.csv")
        if not os.path.isfile(candidate):
            matches = glob.glob(os.path.join(base_dir, "**", f"{symbol}.csv"), recursive=True)
            if not matches:
                print(f"❌ File not found for {symbol}, skipping.")
                continue
            candidate = matches[0]

        print(f"Reading: {candidate}")
        try:
            df = pd.read_csv(
                candidate,
                skiprows=2,  # Skip Ticker row and Date label row
                header=0,    # Use the "Price, Close, High..." line as headers
                names=["Date", "Close", "High", "Low", "Open", "Volume"],
                parse_dates=["Date"],
            )

            df.set_index("Date", inplace=True)
            df.index.name = "Date"
            dfs[symbol] = df

        except Exception as e:
            print(f"❌ Error reading {symbol}: {e}")
            continue

    return dfs


def multi_horizon_labeling(prices, horizon=3, threshold=0.01, price_col='Close'):
    """
    Label using future price movements over a rolling horizon.
    Accepts Series, ndarray, or DataFrame with price_col specified.
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices[price_col].values
    elif isinstance(prices, pd.Series):
        prices = prices.values
    elif isinstance(prices, np.ndarray):
        pass
    else:
        raise ValueError("Unsupported type for 'prices'")

    n = len(prices)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        current_price = prices[i]
        for j in range(1, horizon + 1):
            future_price = prices[i + j]
            ret = (future_price / current_price) - 1
            if ret > threshold:
                labels[i] = 1
                break
            elif ret < -threshold:
                labels[i] = -1
                break

    buy_indices = np.where(labels == 1)[0]
    sell_indices = np.where(labels == -1)[0]
    return buy_indices, sell_indices



def plot_labels(df, buy_indices, sell_indices, price_col='Close', title=None, max_points=200):
    """
    Plot price chart with triple-barrier buy/sell points.

    Parameters:
    - df: pandas DataFrame with price data indexed by date
    - buy_indices: numpy array of indices where profit target was hit (buy signals)
    - sell_indices: numpy array of indices where stop loss was hit (sell signals)
    - price_col: price column name
    - title: optional title for the plot
    - max_points: max points to plot (for performance)
    """
    df_plot = df.iloc[:max_points].copy()

    # Clip indices to max_points range to avoid indexing errors
    buy_indices = buy_indices[buy_indices < max_points]
    sell_indices = sell_indices[sell_indices < max_points]

    plt.figure(figsize=(14, 7))
    plt.plot(df_plot.index, df_plot[price_col], label='Price', color='blue')

    plt.scatter(df_plot.index[buy_indices], df_plot[price_col].iloc[buy_indices],
                marker='^', color='green', s=100, label='Profit Target Hit (Buy)')
    plt.scatter(df_plot.index[sell_indices], df_plot[price_col].iloc[sell_indices],
                marker='v', color='red', s=100, label='Stop Loss Hit (Sell)')

    plt.title(title or 'Triple Barrier Buy/Sell Points')
    plt.xlabel('Date')
    plt.ylabel(price_col)
    plt.legend()
    plt.grid(True)
    plt.show()

def expand_indices(peaks, tolerance):
    expanded = set()
    for idx in peaks:
        for offset in range(-tolerance, tolerance + 1):
            new_idx = idx + offset
            if new_idx >= 0:  # Only avoid negative indices
                expanded.add(new_idx)
    return sorted(expanded)

def add_technical_indicators(df):
    df = df.copy()

    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    df.ta.bbands(length=20, std=2, append=True)
    df.rename(columns={'BBU_20_2.0': 'BB_upper', 'BBL_20_2.0': 'BB_lower'}, inplace=True)

    # MACD
    df.ta.macd(close='Close', append=True)
    df.rename(columns={
        'MACD_12_26_9': 'MACD',
        'MACDs_12_26_9': 'MACD_signal'
    }, inplace=True)

    # RSI, ATR
    df.ta.rsi(length=14, append=True)
    df.rename(columns={'RSI_14': 'RSI_14'}, inplace=True)
    df.ta.atr(length=14, append=True)
    df.rename(columns={'ATRr_14': 'ATR_14'}, inplace=True)

    # Other indicators
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.cci(length=20, append=True)
    
    # FIXED: Rolling OBV instead of cumulative
    obv_change = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                 np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    df['OBV_20'] = pd.Series(obv_change, index=df.index).rolling(window=20).sum()
    
    df.ta.cmf(length=20, append=True)
    df.ta.adx(append=True)

    # Log returns & Volatility
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_20'] = df['LogRet'].rolling(window=20).std() * np.sqrt(252)

    # Time-series features
    df['zscore_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['skew_20'] = df['Close'].rolling(window=20).skew()
    df['kurt_20'] = df['Close'].rolling(window=20).kurt()
    df.ta.roc(length=10, append=True)
    
    # Derived price features
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # FIXED: Rolling VWAP instead of cumulative
    vwap_window = 20
    df['VWAP'] = (
        (df['Typical_Price'] * df['Volume']).rolling(window=vwap_window).sum() / 
        df['Volume'].rolling(window=vwap_window).sum()
    )
    
    df['HL2'] = (df['High'] + df['Low']) / 2
    df['OHLC4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Final cleanup
    df.dropna(inplace=True)
    return df

def detect_and_plot_price_movements(prices, lookforward=5, up_threshold=0.10, down_threshold=-0.10, plot=True):
    """
    Detects future up/down movements by looking forward N days.
    Labels are assigned if threshold is reached AT ANY POINT within the window.
    
    - Label 2 ("Buy"):  price will reach >= up_threshold ANYWHERE in next lookforward days
    - Label 1 ("Sell"): price will reach <= down_threshold ANYWHERE in next lookforward days
    - Label 0 ("Hold"): neither condition met within window
    
    If BOTH thresholds are hit within the window, the FIRST one reached is used.
    
    Parameters
    ----------
    prices : list or array
        Price series.
    lookforward : int
        Number of days to look ahead (default 5).
    up_threshold : float
        Percent threshold for buy signal (default 0.10 = 10%).
    down_threshold : float
        Percent threshold for sell signal (default -0.10 = -10%).
    plot : bool
        Whether to show the plot.
    
    Returns
    -------
    dict with:
        - labels: np.array of int (0=Hold, 1=Sell, 2=Buy)
        - days_to_target: days until target reached (np.nan if not reached)
        - peak_gain: maximum gain % reached in window
        - max_loss: maximum loss % reached in window
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    
    if n < 2:
        return {
            'labels': np.array([], dtype=int),
            'days_to_target': np.array([]),
            'peak_gain': np.array([]),
            'max_loss': np.array([])
        }
    
    labels = np.zeros(n, dtype=int)
    days_to_target = np.full(n, np.nan)
    peak_gain = np.zeros(n)
    max_loss = np.zeros(n)
    
    for i in range(n):
        # Look forward up to lookforward days (or until end of data)
        end_idx = min(i + lookforward + 1, n)
        future_window = prices[i+1:end_idx]  # Exclude current price
        
        if len(future_window) == 0:
            # Not enough future data, label as Hold
            labels[i] = 0
            continue
        
        # Calculate percent changes from current price to all future prices
        pct_changes = (future_window - prices[i]) / prices[i]
        
        # Track peak gain and max loss in window
        peak_gain[i] = np.max(pct_changes)
        max_loss[i] = np.min(pct_changes)
        
        # Find first day each threshold is crossed
        up_cross_days = np.where(pct_changes >= up_threshold)[0]
        down_cross_days = np.where(pct_changes <= down_threshold)[0]
        
        first_up = up_cross_days[0] + 1 if len(up_cross_days) > 0 else np.inf
        first_down = down_cross_days[0] + 1 if len(down_cross_days) > 0 else np.inf
        
        # Label based on which threshold is reached first
        if first_up < first_down:
            labels[i] = 2  # Buy signal (up target reached first)
            days_to_target[i] = first_up
        elif first_down < first_up:
            labels[i] = 1  # Sell signal (down target reached first)
            days_to_target[i] = first_down
        else:
            labels[i] = 0  # Hold (neither reached)
    
    # Optional plotting
    if plot:
        time = np.arange(len(prices))
        fig = plt.figure(figsize=(16, 12))
        
        # Subplot 1: Price with signals
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(time, prices, label="Price", linewidth=2, color='black', alpha=0.7)
        
        # Plot Buy signals with color intensity based on days to target
        buy_idxs = np.where(labels == 2)[0]
        if len(buy_idxs) > 0:
            # Color by speed: faster = darker green
            buy_days = days_to_target[buy_idxs]
            buy_colors = plt.cm.Greens(1 - (buy_days - 1) / lookforward)
            ax1.scatter(buy_idxs, prices[buy_idxs], c=buy_colors, marker="^", 
                       s=150, label=f"Buy (≥{up_threshold*100:.0f}% within {lookforward}d)", 
                       zorder=3, edgecolors='darkgreen', linewidths=2)
            
            # Draw arrows showing when target is reached
            for idx in buy_idxs[:10]:  # Limit to first 10 for clarity
                target_day = int(idx + days_to_target[idx])
                if target_day < len(prices):
                    ax1.annotate('', xy=(target_day, prices[target_day]), 
                               xytext=(idx, prices[idx]),
                               arrowprops=dict(arrowstyle='->', color='green', 
                                             alpha=0.3, linewidth=1.5))
        
        # Plot Sell signals
        sell_idxs = np.where(labels == 1)[0]
        if len(sell_idxs) > 0:
            sell_days = days_to_target[sell_idxs]
            sell_colors = plt.cm.Reds(1 - (sell_days - 1) / lookforward)
            ax1.scatter(sell_idxs, prices[sell_idxs], c=sell_colors, marker="v", 
                       s=150, label=f"Sell (≤{down_threshold*100:.0f}% within {lookforward}d)", 
                       zorder=3, edgecolors='darkred', linewidths=2)
            
            for idx in sell_idxs[:10]:
                target_day = int(idx + days_to_target[idx])
                if target_day < len(prices):
                    ax1.annotate('', xy=(target_day, prices[target_day]), 
                               xytext=(idx, prices[idx]),
                               arrowprops=dict(arrowstyle='->', color='red', 
                                             alpha=0.3, linewidth=1.5))
        
        ax1.set_title(f"Price Movements with {lookforward}-Day Lookahead (Target Reached Anywhere in Window)", 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price ($)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Subplot 2: Days to target histogram
        ax2 = plt.subplot(4, 1, 2)
        valid_days = days_to_target[~np.isnan(days_to_target)]
        if len(valid_days) > 0:
            buy_days_valid = days_to_target[buy_idxs]
            sell_days_valid = days_to_target[sell_idxs]
            
            bins = np.arange(1, lookforward + 2) - 0.5
            ax2.hist(buy_days_valid, bins=bins, alpha=0.6, color='green', 
                    label=f'Buy signals (avg: {np.mean(buy_days_valid):.1f}d)', edgecolor='black')
            ax2.hist(sell_days_valid, bins=bins, alpha=0.6, color='red', 
                    label=f'Sell signals (avg: {np.mean(sell_days_valid):.1f}d)', edgecolor='black')
            ax2.set_xlabel('Days Until Target Reached')
            ax2.set_ylabel('Count')
            ax2.set_title('Distribution of Days to Reach Target', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_xticks(range(1, lookforward + 1))
        
        # Subplot 3: Peak gains and max losses
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(time, peak_gain * 100, 'g-', alpha=0.6, linewidth=1.5, label='Peak Gain in Window')
        ax3.plot(time, max_loss * 100, 'r-', alpha=0.6, linewidth=1.5, label='Max Loss in Window')
        ax3.axhline(up_threshold * 100, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
        ax3.axhline(down_threshold * 100, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax3.set_ylabel('Return (%)')
        ax3.set_title(f'Peak Returns Within {lookforward}-Day Window', fontsize=12)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Label distribution
        ax4 = plt.subplot(4, 1, 4)
        label_counts = [np.sum(labels == 0), np.sum(labels == 1), np.sum(labels == 2)]
        label_names = ['Hold', 'Sell', 'Buy']
        colors = ['gray', 'red', 'green']
        
        bars = ax4.bar(label_names, label_counts, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Count')
        ax4.set_title('Label Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add counts and percentages on bars
        for bar, count in zip(bars, label_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}\n({count/len(labels)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        print("\n" + "="*70)
        print("LOOKAHEAD LABELING STATISTICS (TARGET REACHED ANYWHERE IN WINDOW)")
        print("="*70)
        print(f"Lookforward period: {lookforward} days")
        print(f"Buy threshold:  ≥{up_threshold*100:+.1f}%")
        print(f"Sell threshold: ≤{down_threshold*100:.1f}%")
        
        print(f"\n{'='*70}")
        print("LABEL DISTRIBUTION:")
        print(f"{'='*70}")
        print(f"Total data points:  {len(labels):5d}")
        print(f"Buy signals (2):    {label_counts[2]:5d} ({label_counts[2]/len(labels)*100:5.1f}%)")
        print(f"Sell signals (1):   {label_counts[1]:5d} ({label_counts[1]/len(labels)*100:5.1f}%)")
        print(f"Hold signals (0):   {label_counts[0]:5d} ({label_counts[0]/len(labels)*100:5.1f}%)")
        
        if len(buy_idxs) > 0:
            print(f"\n{'='*70}")
            print("BUY SIGNAL ANALYSIS:")
            print(f"{'='*70}")
            print(f"Average days to target:     {np.mean(days_to_target[buy_idxs]):.2f}")
            print(f"Median days to target:      {np.median(days_to_target[buy_idxs]):.2f}")
            print(f"Fastest target reached:     {np.min(days_to_target[buy_idxs]):.0f} day(s)")
            print(f"Slowest target reached:     {np.max(days_to_target[buy_idxs]):.0f} day(s)")
            print(f"Average peak gain:          {np.mean(peak_gain[buy_idxs])*100:.2f}%")
            
            # Days distribution
            for day in range(1, lookforward + 1):
                count = np.sum(days_to_target[buy_idxs] == day)
                if count > 0:
                    print(f"  Day {day}: {count:3d} signals ({count/len(buy_idxs)*100:5.1f}%)")
        
        if len(sell_idxs) > 0:
            print(f"\n{'='*70}")
            print("SELL SIGNAL ANALYSIS:")
            print(f"{'='*70}")
            print(f"Average days to target:     {np.mean(days_to_target[sell_idxs]):.2f}")
            print(f"Median days to target:      {np.median(days_to_target[sell_idxs]):.2f}")
            print(f"Fastest target reached:     {np.min(days_to_target[sell_idxs]):.0f} day(s)")
            print(f"Slowest target reached:     {np.max(days_to_target[sell_idxs]):.0f} day(s)")
            print(f"Average max loss:           {np.mean(max_loss[sell_idxs])*100:.2f}%")
            
            # Days distribution
            for day in range(1, lookforward + 1):
                count = np.sum(days_to_target[sell_idxs] == day)
                if count > 0:
                    print(f"  Day {day}: {count:3d} signals ({count/len(sell_idxs)*100:5.1f}%)")
        
        # Conflict analysis (both thresholds hit)
        both_hit = (peak_gain >= up_threshold) & (max_loss <= down_threshold)
        if np.any(both_hit):
            print(f"\n{'='*70}")
            print("CONFLICT ANALYSIS (Both thresholds hit in same window):")
            print(f"{'='*70}")
            print(f"Conflicts detected:         {np.sum(both_hit):5d} ({np.sum(both_hit)/len(labels)*100:5.1f}%)")
            conflict_labels = labels[both_hit]
            print(f"  Labeled as Buy:           {np.sum(conflict_labels == 2):5d}")
            print(f"  Labeled as Sell:          {np.sum(conflict_labels == 1):5d}")
            print("(Label determined by which threshold was reached FIRST)")
        
        print("="*70 + "\n")
    
    return {
        'labels': labels,
        'days_to_target': days_to_target,
        'peak_gain': peak_gain,
        'max_loss': max_loss
    }

import shutil

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory and contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def process_windows(processed_dfs, days, name="run", symbol_names=None):
    fft_features = ['Close', 'High', 'Low', 'Volume', 'Typical_Price', 'VWAP', 'HL2', 'OHLC4', 'MA20']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    subdir = "daily_data/" if name == "daily" else ""
    prefix = f"train-val-data/{subdir}"
    if name == 'daily':
        clear_folder(prefix)
        
    raw_data_path = f"{prefix}data_windows_{name}_{timestamp}.npy"
    raw_label_path = f"{prefix}labels_{name}_{timestamp}.npy"
    symbol_path = f"{prefix}symbols_{name}_{timestamp}.npy"
    
    total_samples = sum(max(0, len(df) - days) for df in processed_dfs)
    first_df = next((df for df in processed_dfs if len(df) > days), None)
    if first_df is None:
        print("No valid dataframes with enough samples")
        return

    dummy_sample = first_df.drop(columns=['Date'], errors='ignore').to_numpy(dtype=np.float32)
    input_dim = dummy_sample.shape[1]
    fft_features_present = [f for f in fft_features if f in first_df.columns]
    feature_dim = input_dim + 3 * len(fft_features_present)

    data_memmap = np.lib.format.open_memmap(
        raw_data_path, dtype='float32', mode='w+', shape=(total_samples, days, feature_dim)
    )
    labels_memmap = np.lib.format.open_memmap(
        raw_label_path, dtype='uint8', mode='w+', shape=(total_samples,)
    )
    symbol_names_array = np.empty(total_samples, dtype=object)
    sample_index = 0

    for idx, df in tqdm(enumerate(processed_dfs), total=len(processed_dfs), desc="Processing symbols"):
        if 'Close' not in df.columns:
            continue
        
        symbol = symbol_names[idx] if symbol_names else f"Symbol_{idx}"
        df_clean = df.drop(columns=['Date'], errors='ignore')
        df_clean[df_clean.select_dtypes(include='bool').columns] = \
            df_clean.select_dtypes(include='bool').astype(int)

        if df_clean.empty or len(df_clean) <= days:
            continue

        if not all(f in df_clean.columns for f in fft_features_present):
            continue

        fft_indices = [df_clean.columns.get_loc(col) for col in fft_features_present]
        data_array = df_clean.to_numpy(dtype=np.float32)

        df_tmp = df['Close'].values #TODO use adjusted close
        plot = False
        #SELL-1 BUY-2
        res = detect_and_plot_price_movements(df_tmp, plot=plot)
        labels = res['labels']
        
        for ti in range(days, len(df)):
            window = data_array[ti - days:ti]
            min_vals = np.nanmin(window, axis=0)
            max_vals = np.nanmax(window, axis=0)
            denom = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
            window_norm = (window - min_vals) / denom
            window_norm = np.nan_to_num(window_norm, nan=0.0)

            fft_input = window_norm[:, fft_indices]
            fft_data = np.fft.fft(fft_input, axis=0)
            fft_mag = np.abs(fft_data)
            fft_phase = np.angle(fft_data)
            fft_mag_norm = (fft_mag - fft_mag.min()) / (fft_mag.max() - fft_mag.min() + 1e-8)
            fft_phase_norm = (fft_phase - fft_phase.min()) / (fft_phase.max() - fft_phase.min() + 1e-8)

            wavelet_features = []
            for col in range(fft_input.shape[1]):
                swt_coeffs = pywt.swt(fft_input[:, col], 'db1', level=1)
                cA = swt_coeffs[0][0]
                wavelet_features.append(cA)
            wavelet_array = np.stack(wavelet_features, axis=-1)
            wavelet_array = (wavelet_array - wavelet_array.min()) / (wavelet_array.max() - wavelet_array.min() + 1e-8)

            pad_len = days - wavelet_array.shape[0]
            if pad_len > 0:
                wavelet_array = np.pad(wavelet_array, ((0, pad_len), (0, 0)), mode='constant')

            combined = np.concatenate([window_norm, fft_mag_norm, fft_phase_norm, wavelet_array], axis=1)
            if combined.shape != (days, feature_dim):
                continue

            data_memmap[sample_index] = combined
            symbol_names_array[sample_index] = symbol
            
            if ti < len(labels):
                labels_memmap[sample_index] = labels[ti]
                sample_index += 1
            else:
                sample_index += 1
                continue  # prevent IndexError if ti+1 is out of bounds


        del df, df_clean, data_array, df_tmp, labels
        gc.collect()
    
    np.save(symbol_path, symbol_names_array[:sample_index])
    print(f"Saved raw data: {sample_index} samples")
    print(f" - Features: {raw_data_path}")
    print(f" - Labels: {raw_label_path}")
    print(f" - Symbols: {symbol_path}")
    
    return raw_data_path, raw_label_path, symbol_path

MIN_ROWS_REQUIRED = 60
def extract_features_with_fft(symbol_list, directory, saveData, name, days_to_process, doBalance=True):
    
    dataframes_list = get_df_list(symbol_list, directory)
        
    # Apply all indicators and enhancements
    processed_dfs = []
    symbols = []
    if not days_to_process:
        print("Processing the whole dataset")
    else:
        print(f"Processing only the last {days_to_process} days")
        
    for i, (symbol, df) in enumerate(dataframes_list.items(), 1):
        print(f"[{i}/{len(dataframes_list)}] Processing: {symbol}")
        df = df.copy()
        
        if days_to_process:
            # Keep only the last `days` rows
            df = df.tail(days_to_process)
    
        if len(df) < MIN_ROWS_REQUIRED:
            print(f"Skipping {symbol}: not enough data ({len(df)} rows)")
            continue  # Skip this symbol and move to the next

        # Dollar volume and percent change
        df["dollar_volume"] = df["Close"] * df["Volume"]
        df["dollar_volume_pct"] = df["dollar_volume"].pct_change().fillna(0)

        # Add temporal features
        df = add_temporal_features(df)

        # Add technical indicators (MA, Bollinger, MACD)
        try:
            df = add_technical_indicators(df)
        except Exception as e:
            print(f"[SKIP] {symbol}: error during indicator processing: {e}")
            continue
        
        # Add advanced features
        #df = advanced_indicators.add_advanced_features(df)
        
        # Check again after indicators are added
        if len(df) < MIN_ROWS_REQUIRED:
            print(f"[SKIP] {symbol}: not enough data after feature extraction")
            continue

        processed_dfs.append(df)
        symbols.append(symbol)
    
    window_days = 60;
    result = process_windows(processed_dfs, window_days, name, symbol_names=symbols)

    if result is None:
        print(f"[Warning] Skipping processing for {symbols} — not enough valid data after filtering.")
        return None
    else:
        raw_data_path, raw_label_path, symbol_path = result
        
    balanced_data, balanced_labels, balanced_symbols, data, labels, symbols = balance_and_save(raw_data_path, raw_label_path, symbol_path, name, doBalance)

    if doBalance:
        return balanced_data, balanced_labels, balanced_symbols
    else:
        return data, labels, symbols

def balance_and_save(raw_data_path, raw_label_path, symbol_path=None, name='default', doBalance=True):
    print("Balancing dataset...")

    data = np.load(raw_data_path, mmap_mode='r')
    labels = np.load(raw_label_path)
    balanced_data = []
    balanced_labels = []
    balanced_symbols = []
    
    if symbol_path:
        symbols = np.load(symbol_path, allow_pickle=True)
    
    if doBalance:
        class_0_idx = np.where(labels == 0)[0]
        class_1_idx = np.where(labels == 1)[0]
        class_2_idx = np.where(labels == 2)[0]
    
        min_class_size = min(len(class_0_idx), len(class_1_idx), len(class_2_idx))
        print(f"Class sizes before balancing: 0={len(class_0_idx)}, 1={len(class_1_idx)}, 2={len(class_2_idx)}")
        print(f"Balancing to {min_class_size} samples per class")
    
        np.random.seed(42)
        balanced_indices = np.concatenate([
            np.random.choice(class_0_idx, min_class_size, replace=False),
            np.random.choice(class_1_idx, min_class_size, replace=False),
            np.random.choice(class_2_idx, min_class_size, replace=False)
        ])
        np.random.shuffle(balanced_indices)
    
        balanced_data = data[balanced_indices]
        balanced_labels = labels[balanced_indices]
    
        # Save balanced data
        subdir = "daily_data/" if name == "daily" else ""
        prefix = f"train-val-data/{subdir}"
        
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_path = f"{prefix}{name}_balanced_data_{ts}.npy"
        label_path = f"{prefix}{name}_balanced_labels_{ts}.npy"
        symbol_balanced_path = None
    
        np.save(data_path, balanced_data)
        np.save(label_path, balanced_labels)
    
        if symbol_path:
            balanced_symbols = symbols[balanced_indices]
            symbol_balanced_path = f"train-val-data/{name}_balanced_symbols_{ts}.npy"
            np.save(symbol_balanced_path, balanced_symbols)
            print(f"Saved balanced symbols: {symbol_balanced_path}")
    
        print("Balanced data saved:")
        print(f" - Features: {data_path}")
        print(f" - Labels: {label_path}")
    
    return balanced_data, balanced_labels, balanced_symbols, data, labels, symbols


def remove_invalid_samples(data_list, label_list):
    clean_data = []
    clean_labels = []
    for x, y in zip(data_list, label_list):
        if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
            clean_data.append(x)
            clean_labels.append(y)
    return clean_data, clean_labels