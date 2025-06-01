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
    
def detect_local_extrema_labels(
    prices,
    gain_threshold=0.05,
    min_distance=1,
    smooth=True,
    ma_window=20,
    plot=False
):
    """
    Detects local minima and maxima, then labels them alternately as buy/sell points.
    Keeps only the ones with gain above threshold.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    if prices is None or len(prices) == 0 or np.all(np.isnan(prices)):
        print("Invalid price array.")
        return [], []

    prices = np.asarray(prices, dtype=np.float32)

    if smooth and len(prices) >= ma_window:
        prices = np.convolve(prices, np.ones(ma_window) / ma_window, mode='same')

    # Detect local maxima and minima
    peak_prominence = 0.005 * prices.max()
    sell_idxs, _ = find_peaks(prices, distance=min_distance, prominence=peak_prominence)
    buy_idxs, _ = find_peaks(-prices, distance=min_distance, prominence=peak_prominence)
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Price')
        plt.plot(buy_idxs, prices[buy_idxs], 'g^', label='Buy', markersize=10)
        plt.plot(sell_idxs, prices[sell_idxs], 'rv', label='Sell', markersize=10)
        plt.legend()
        plt.title('Buy/Sell Points Before Filtering')
        plt.show()
        
    extrema = sorted(
        [(idx, 'buy') for idx in buy_idxs] +
        [(idx, 'sell') for idx in sell_idxs],
        key=lambda x: x[0]
    )

    # Alternate: buy -> sell -> buy -> ...
    filtered_buys = []
    filtered_sells = []

    i = 0
    while i < len(extrema) - 1:
        curr_idx, curr_type = extrema[i]
        next_idx, next_type = extrema[i + 1]

        if curr_type == 'buy' and next_type == 'sell':
            gain = (prices[next_idx] - prices[curr_idx]) / prices[curr_idx]
            if gain >= gain_threshold:
                filtered_buys.append(curr_idx)
                filtered_sells.append(next_idx)
            i += 2  # Skip the pair
        else:
            i += 1  # Skip unmatched or out-of-sequence points

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(prices, label="Price")
        plt.plot(filtered_buys, prices[filtered_buys], 'g^', label="Buy", markersize=10)
        plt.plot(filtered_sells, prices[filtered_sells], 'rv', label="Sell", markersize=10)
        plt.title("Local Minima/Maxima with Gain Threshold")
        plt.legend()
        plt.show()

    return np.array(filtered_buys, dtype=int), np.array(filtered_sells, dtype=int)


def detect_labels_via_peaks(prices, gain_threshold=0.1,
                            min_distance=1, smooth=True, ma_window=20, plot=False):
    """
    Detect sell (local maxima) and buy (local minima) points where the gain exceeds the threshold.
    A pair is kept if:
        - gain from a buy to its next sell > threshold, OR
        - gain from a sell to its next buy > threshold
    """
    # Ensure prices is not empty or NaN
    if prices is None or len(prices) == 0 or np.all(np.isnan(prices)):
        print("Error: Empty or NaN array passed to detect_labels_via_peaks.")
        return [], []

    prices = np.asarray(prices, dtype=np.float32)

    if smooth and len(prices) >= ma_window:
        prices = np.convolve(prices, np.ones(ma_window) / ma_window, mode='same')

    # Step 1: Find peaks
    prominence = 0.005 * prices.max()
    sell_idxs, _ = scipy.signal.find_peaks(prices, distance=min_distance, prominence=prominence)
    buy_idxs, _ = scipy.signal.find_peaks(-prices, distance=min_distance, prominence=prominence)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Price')
        plt.plot(buy_idxs, prices[buy_idxs], 'g^', label='Buy', markersize=10)
        plt.plot(sell_idxs, prices[sell_idxs], 'rv', label='Sell', markersize=10)
        plt.legend()
        plt.title('Buy/Sell Points Before Filtering')
        plt.show()

    # Step 2: Filter pairs based on gain
    valid_buy_idxs = []
    valid_sell_idxs = []

    # Sort for chronological order
    buy_idxs = np.sort(buy_idxs)
    sell_idxs = np.sort(sell_idxs)

    i, j = 0, 0
    while i < len(buy_idxs) and j < len(sell_idxs):
        buy_idx = buy_idxs[i]
        sell_idx = sell_idxs[j]

        if sell_idx <= buy_idx:
            j += 1  # Sell must come after buy
            continue

        gain_buy_to_sell = (prices[sell_idx] - prices[buy_idx]) / prices[buy_idx]

        # Look ahead for next buy after current sell
        next_buy_idx = next((b for b in buy_idxs if b > sell_idx), None)
        gain_sell_to_buy = (prices[next_buy_idx] - prices[sell_idx]) / prices[sell_idx] if next_buy_idx else -np.inf

        if gain_buy_to_sell >= gain_threshold or gain_sell_to_buy >= gain_threshold:
            valid_buy_idxs.append(buy_idx)
            valid_sell_idxs.append(sell_idx)
            i += 1
            j += 1
        else:
            i += 1  # Move to next buy, keep sell for future matching

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Price')
        plt.plot(valid_buy_idxs, prices[valid_buy_idxs], 'g^', label='Valid Buy', markersize=10)
        plt.plot(valid_sell_idxs, prices[valid_sell_idxs], 'rv', label='Valid Sell', markersize=10)
        plt.legend()
        plt.title('Filtered Buy/Sell Points with Gain Threshold')
        plt.show()

    return np.array(valid_buy_idxs, dtype=int), np.array(valid_sell_idxs, dtype=int)

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
    df.ta.macd(append=True)
    df.rename(columns={
        'MACD_12_26_9': 'MACD',
        'MACDs_12_26_9': 'MACD_signal'
    }, inplace=True)

    # RSI, ATR
    df.ta.rsi(length=14, append=True)
    df.rename(columns={'RSI_14': 'RSI_14'}, inplace=True)
    df.ta.atr(length=14, append=True)
    df.rename(columns={'ATRr_14': 'ATR_14'}, inplace=True)

    # 🔼 NEW INDICATORS
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.obv(append=True)
    df.ta.cmf(length=20, append=True)
    df.ta.adx(append=True)

    # Log returns & Volatility
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_20'] = df['LogRet'].rolling(window=20).std() * np.sqrt(252)

    # 🔼 TIME-SERIES FEATURES
    df['zscore_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['skew_20'] = df['Close'].rolling(window=20).skew()
    df['kurt_20'] = df['Close'].rolling(window=20).kurt()
    df.ta.roc(length=10, append=True)
    
    # ✅ Derived price features
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['HL2'] = (df['High'] + df['Low']) / 2
    df['OHLC4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    # Final cleanup
    df.dropna(inplace=True)
    return df

def process_windows(processed_dfs, days, name="run", symbol_names=None):
    fft_features = ['Close', 'High', 'Low', 'Volume', 'Typical_Price', 'VWAP', 'HL2', 'OHLC4', 'MA20']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    subdir = "daily_data/" if name == "daily" else ""
    prefix = f"train-val-data/{subdir}"

    raw_data_path = f"{prefix}data_windows_{name}_{timestamp}.npy"
    raw_label_path = f"{prefix}labels_{name}_{timestamp}.npy"
    symbol_path = f"{prefix}symbols_{name}_{timestamp}.npy"
    
    total_samples = sum(max(0, len(df) - days) for df in processed_dfs)
    first_df = next((df for df in processed_dfs if len(df) > days), None)
    if first_df is None:
        raise ValueError("No valid dataframes with enough samples")

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

        df_tmp = df['Close'].values
        plot = False
        #buy_peaks, sell_peaks = detect_labels_via_peaks(df_tmp, gain_threshold=0.1, min_distance=1, smooth=True, ma_window=20, plot=plot)
        buy_peaks, sell_peaks = detect_local_extrema_labels(
            df_tmp,
            gain_threshold=0.1,
            min_distance=1,
            smooth=True,
            ma_window=20,
            plot=plot
            )
        # expand indexes with a tolerance of 1
        #buy_peaks = expand_indices(buy_peaks, tolerance=1)
        #sell_peaks = expand_indices(sell_peaks, tolerance=1)

        labels = np.zeros(len(df_tmp), dtype=np.uint8)
        labels[sell_peaks] = 1
        labels[buy_peaks] = 2

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
            labels_memmap[sample_index] = labels[ti]
            symbol_names_array[sample_index] = symbol
            sample_index += 1

        del df, df_clean, data_array, df_tmp, labels
        gc.collect()
    
    np.save(symbol_path, symbol_names_array[:sample_index])
    print(f"Saved raw data: {sample_index} samples")
    print(f" - Features: {raw_data_path}")
    print(f" - Labels: {raw_label_path}")
    print(f" - Symbols: {symbol_path}")
    
    return raw_data_path, raw_label_path, symbol_path

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

        # Dollar volume and percent change
        df["dollar_volume"] = df["Close"] * df["Volume"]
        df["dollar_volume_pct"] = df["dollar_volume"].pct_change().fillna(0)

        # Add temporal features
        df = add_temporal_features(df)

        # Add technical indicators (MA, Bollinger, MACD)
        df = add_technical_indicators(df)
        
        # Add advanced features
        df = advanced_indicators.add_advanced_features(df)

        processed_dfs.append(df)
        symbols.append(symbol)
    
    window_days = 30;
    raw_data_path, raw_label_path, symbol_path = process_windows(processed_dfs, window_days, name, symbol_names=symbols)
    balanced_data, balanced_labels, balanced_symbols, data, labels, symbols = balance_and_save(raw_data_path, raw_label_path, symbol_path, name)
    
    if doBalance:
        return balanced_data, balanced_labels, balanced_symbols
    else:
        return data, labels, symbols

def balance_and_save(raw_data_path, raw_label_path, symbol_path=None, name='default'):
    print("Balancing dataset...")

    data = np.load(raw_data_path, mmap_mode='r')
    labels = np.load(raw_label_path)

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
        symbols = np.load(symbol_path, allow_pickle=True)
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