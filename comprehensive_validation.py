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
import matplotlib

#matplotlib.use('Agg')  # Use non-interactive backend
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

import warnings

import time
import requests
from pathlib import Path

CACHE_PATH = Path.home() / ".cache" / "sp500_tickers.csv"
CACHE_TTL = 24 * 3600  # seconds — refresh once per day

def add_relaxed_trend_filter(prices, signal_mask, signal_type='buy', 
                             short_ma=20, long_ma=50):
    """Relaxed: just check MA crossover, no slope requirement."""
    s = pd.Series(prices)
    ma_short = s.rolling(short_ma, min_periods=1).mean()
    ma_long = s.rolling(long_ma, min_periods=1).mean()
    
    if signal_type == 'buy':
        trend_ok = ma_short > ma_long
    else:  # sell
        trend_ok = ma_short < ma_long
    
    return signal_mask & trend_ok.fillna(False).to_numpy()

def require_price_below_ma(prices, signal_mask, ma_period=200):
    """Only buy when price is above long-term MA (bull market filter)."""
    s = pd.Series(prices)
    ma_long = s.rolling(ma_period, min_periods=1).mean()
    above_ma = s < ma_long
    return signal_mask & above_ma.fillna(False).to_numpy()

def fetch_sp500_from_wikipedia():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    df = pd.read_html(r.text, displayed_only=False)[0]
    symbols = [s.replace('.', '-') for s in df['Symbol'].astype(str).tolist()]
    return symbols

def read_sp500_from_cache():
    if not CACHE_PATH.exists():
        return None
    mtime = CACHE_PATH.stat().st_mtime
    if time.time() - mtime > CACHE_TTL:
        return None
    df = pd.read_csv(CACHE_PATH)
    return df['symbol'].astype(str).tolist()

def write_sp500_cache(symbols):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(CACHE_PATH, index=False)

def get_sp500_tickers():
    # 1) try cache
    symbols = read_sp500_from_cache()
    if symbols:
        return symbols

    # 2) try wikipedia (with headers)
    try:
        symbols = fetch_sp500_from_wikipedia()
        write_sp500_cache(symbols)
        return symbols
    except Exception as e:
        # log but continue to fallback
        print(f"Warning: failed to fetch S&P 500 from Wikipedia: {e}")

    # 3) last-resort fallback: if you want, load a local static file shipped with your repo
    local = Path("data/sp500_static.csv")
    if local.exists():
        df = pd.read_csv(local)
        return [s.replace('.', '-') for s in df['symbol'].astype(str).tolist()]

    raise RuntimeError("Could not obtain S&P 500 tickers (no cache, wikipedia fetch failed, no local fallback).")

def filter_sp500_tickers(tickers):
    sp500 = set(get_sp500_tickers())
    return [t for t in tickers if t in sp500]

def directional_confidence_signals(pred_test, trend_window=3, conf_th=0.0,
                                   smooth_window=2, window_offset=0):
    """
    Directional signals using smoothed confidences (no future info).
    BUY if class2 is a local peak AND (class1 AND class0) is a local trough.
    SELL if class1 is a local peak AND (class2 AND class0) is a local trough.
    
    Args:
        pred_test: ndarray (n, n_classes) with at least 3 classes (0,1,2).
        trend_window: lookback window for peak/trough check.
        conf_th: minimum confidence required for signal.
        window_offset: exclude last n elements from window (default 1).
                      offset=1 means compare current value to window ending 1 step back.
    
    Returns:
        dict with buy_mask, sell_mask, indices, strengths, and details.
    """
    pred_test = np.asarray(pred_test)
    if pred_test.ndim != 2 or pred_test.shape[1] < 3:
        raise ValueError("pred_test must be shape (n, ≥3 classes)")
    
    n = pred_test.shape[0]
    c0, c1, c2 = pred_test[:, 0], pred_test[:, 1], pred_test[:, 2]
    
    c0 = pd.DataFrame(c0).rolling(smooth_window, min_periods=1).mean().to_numpy()
    c1 = pd.DataFrame(c1).rolling(smooth_window, min_periods=1).mean().to_numpy()
    c2 = pd.DataFrame(c2).rolling(smooth_window, min_periods=1).mean().to_numpy()
    
    buy_mask = np.zeros(n, dtype=bool)
    sell_mask = np.zeros(n, dtype=bool)
    buy_strength = np.zeros(n)
    sell_strength = np.zeros(n)
    
    trend_window = max(1, int(trend_window))
    conf_th = float(conf_th)
    window_offset = max(1, int(window_offset))
    
    for t in range(n):
        # Need enough history for offset window
        if t < window_offset:
            continue
            
        # Window ends at (t - window_offset), goes back trend_window steps
        end = t - window_offset + 1  # +1 because slicing is exclusive at end
        start = max(0, end - trend_window)
        
        c0_win = c0[start:end]
        c1_win = c1[start:end]
        c2_win = c2[start:end]
        
        # Current values at time t
        c0_t, c1_t, c2_t = c0[t], c1[t], c2[t]
        
        if (c1_t < conf_th) and (c2_t < conf_th):
            continue
        
        # Check if window has data
        if len(c0_win) == 0:
            continue
        
        # Peak/trough: compare current value to offset window
        c0_is_trough = c0_t <= np.min(c0_win)
        c1_is_peak = c1_t >= np.max(c1_win)
        c1_is_trough = c1_t <= np.min(c1_win)
        c2_is_peak = c2_t >= np.max(c2_win)
        c2_is_trough = c2_t <= np.min(c2_win)
        
        # BUY: class2 peak and (class1 AND class0 trough)
        if c2_is_peak and c1_is_trough and c0_is_trough and c2_t >= conf_th  and c1_t <= 0.001:
            buy_mask[t] = True
            buy_strength[t] = c2_t - np.min(c2_win)
        
        # SELL: class1 peak and (class2 AND class0 trough)
        if c1_is_peak and c2_is_trough and c0_is_trough and c1_t >= conf_th and c2_t <= 0.001:
            sell_mask[t] = True
            sell_strength[t] = c1_t - np.min(c1_win)
    
    return {
        "buy_mask": buy_mask,
        "sell_mask": sell_mask,
        "buy_idx": np.where(buy_mask)[0],
        "sell_idx": np.where(sell_mask)[0],
        "buy_strength": buy_strength,
        "sell_strength": sell_strength,
        "details": {
            "trend_window": trend_window,
            "conf_th": conf_th,
            "window_offset": window_offset,
            "buy_class": 2,
            "sell_class": 1,
        },
    }



def find_high_confidence_clusters(confidences, pred_classes, target_class, 
                                   conf_threshold=0.95, min_cluster_size=5, 
                                   last_n_growing=5, proximity_pct=0.90):
    """
    Find clusters of high confidence predictions for a given class.
    
    Args:
        confidences (np.ndarray): Confidence values for all predictions
        pred_classes (np.ndarray): Predicted class labels (0 or 1)
        target_class (int): Class to look for (0=sell, 1=buy)
        conf_threshold (float): Minimum confidence threshold (default 0.95)
        min_cluster_size (int): Minimum number of elements in a cluster (default 5)
        last_n_growing (int): Number of last elements that must show growing confidence (default 5)
        proximity_pct (float): Last element must be within this % of current position (default 0.90)
    
    Returns:
        np.ndarray: Boolean mask indicating valid signal positions
    """
    n = len(confidences)
    signal_mask = np.zeros(n, dtype=bool)
    
    # Process each position i
    for i in range(n):
        # Look back from position i
        window_start = max(0, int(i * (1 - proximity_pct)))
        
        # Find high confidence predictions of target class in the lookback window
        candidates = []
        for j in range(window_start, i):
            if pred_classes[j] == target_class and confidences[j] >= conf_threshold:
                candidates.append(j)
        
        # Check if we have at least min_cluster_size candidates
        if len(candidates) < min_cluster_size:
            continue
        
        # Take the last min_cluster_size candidates to form the cluster
        cluster_indices = candidates[-min_cluster_size:]
        cluster_confidences = confidences[cluster_indices]
        
        # Check if last last_n_growing elements show growing confidence
        if len(cluster_indices) >= last_n_growing:
            last_n_conf = cluster_confidences[-last_n_growing:]
            # Check if confidences are strictly increasing (or non-decreasing)
            if np.all(np.diff(last_n_conf) >= 0):  # Use > 0 for strictly increasing
                signal_mask[i] = True
    
    return signal_mask


def ma_trend_check(prices, short=10, long=200, mode="bull"):
    """
    Simple moving-average trend check (safe for live trading).

    - 'bull' → True where short MA > long MA  (buy conditions)
    - 'bear' → True where short MA < long MA  (sell conditions)
    - Auto-corrects invalid window sizes (short/long < 1 → coerced to 1)

    Args:
        prices (array-like): Series of prices.
        short (int): Short moving average window (>=1).
        long (int): Long moving average window (>=1).
        mode (str): 'bull' or 'bear'.

    Returns:
        np.ndarray[bool]: Boolean mask of same length as prices.
    """
    # Ensure valid numeric input
    prices = np.asarray(prices, dtype=float)
    s = pd.Series(prices)

    # Coerce invalid inputs to safe values
    try:
        short = int(short)
        if short < 1:
            warnings.warn(f"ma_trend_check: short window {short} adjusted to 1", UserWarning)
            short = 1
    except Exception:
        short = 1

    try:
        long = int(long)
        if long < 1:
            warnings.warn(f"ma_trend_check: long window {long} adjusted to 1", UserWarning)
            long = 1
    except Exception:
        long = 1

    # Compute causal moving averages
    ma_short = s.rolling(window=short, min_periods=1).mean()
    ma_long = s.rolling(window=long, min_periods=1).mean()

    # Compare MAs
    if mode == "bull":
        mask = ma_short > ma_long # INVERTED LOGIC OTHERWISE IS TOO LATE
    elif mode == "bear":
        mask = ma_short < ma_long
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return mask.to_numpy(dtype=bool)


def strict_rolling_extrema(prices, lookback=5, threshold=0.00, mode="min"):
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

class TradeSimulator:
    """Simulates trades with 5-day hold or 10% gain exit strategy."""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.open_position = None
        
    def execute_trade(self, idx, action, price, dates):
        """
        Execute a buy or sell trade.
        
        Args:
            idx (int): Current index in price array
            action (str): 'buy' or 'sell'
            price (float): Current price
            dates (pd.DatetimeIndex): Date index
        """
        if action == 'buy' and self.open_position is None:
            # Open long position
            shares = self.capital / price
            self.open_position = {
                'type': 'long',
                'entry_idx': idx,
                'entry_date': dates[idx],
                'entry_price': price,
                'shares': shares,
                'days_held': 0
            }
            
        elif action == 'sell' and self.open_position is None:
            # Open short position
            shares = self.capital / price
            self.open_position = {
                'type': 'short',
                'entry_idx': idx,
                'entry_date': dates[idx],
                'entry_price': price,
                'shares': shares,
                'days_held': 0
            }
    
    def check_exit(self, idx, price, dates):
        """
        Check if position should be closed (5 days, 10% gain, or 5% loss).
        
        Args:
            idx (int): Current index
            price (float): Current price
            dates (pd.DatetimeIndex): Date index
            
        Returns:
            bool: True if position was closed
        """
        if self.open_position is None:
            return False
        
        self.open_position['days_held'] += 1
        entry_price = self.open_position['entry_price']
        pos_type = self.open_position['type']
        
        # Calculate gain/loss
        if pos_type == 'long':
            pct_change = (price - entry_price) / entry_price
        else:  # short
            pct_change = (entry_price - price) / entry_price
        
        # Exit conditions: 5 days OR 10% gain OR 5% loss
        should_exit = (
            (self.open_position['days_held'] >= 5) or 
            (pct_change >= 0.10) or 
            (pct_change <= -0.025)
        )
        
        if should_exit:
            # Calculate new capital based on the actual pct_change
            exit_value = self.capital * (1 + pct_change)
            profit = exit_value - self.capital
            
            trade_record = {
                'ticker': self.open_position.get('ticker', 'N/A'),
                'type': pos_type,
                'entry_date': self.open_position['entry_date'],
                'entry_price': entry_price,
                'exit_date': dates[idx],
                'exit_price': price,
                'days_held': self.open_position['days_held'],
                'pct_return': pct_change * 100,
                'profit': profit,
                'capital_after': exit_value,
                'exit_reason': self._get_exit_reason(self.open_position['days_held'], pct_change)
            }
            
            self.trades.append(trade_record)
            self.capital = exit_value
            self.open_position = None
            return True
        
        return False

    def _get_exit_reason(self, days_held, pct_change):
        """Helper to identify why trade was exited."""
        if pct_change >= 0.10:
            return 'take_profit'
        elif pct_change <= -0.05:
            return 'stop_loss'
        elif days_held >= 5:
            return 'time_limit'
        return 'unknown'
    
    def close_final_position(self, price, date):
        """Force close any remaining open position at end of data."""
        if self.open_position is not None:
            entry_price = self.open_position['entry_price']
            pos_type = self.open_position['type']
            
            if pos_type == 'long':
                pct_change = (price - entry_price) / entry_price
            else:
                pct_change = (entry_price - price) / entry_price
            
            exit_value = self.capital * (1 + pct_change)
            profit = exit_value - self.capital
            
            trade_record = {
                'ticker': self.open_position.get('ticker', 'N/A'),
                'type': pos_type,
                'entry_date': self.open_position['entry_date'],
                'entry_price': entry_price,
                'exit_date': date,
                'exit_price': price,
                'days_held': self.open_position['days_held'],
                'pct_return': pct_change * 100,
                'profit': profit,
                'capital_after': exit_value
            }
            
            self.trades.append(trade_record)
            self.capital = exit_value
            self.open_position = None
    
    def get_performance_summary(self):
        """Generate performance statistics."""
        if not self.trades:
            summary = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_return_pct': 0,
                'avg_winning_return': 0,
                'avg_losing_return': 0,
                'total_return_pct': 0,
                'final_capital': self.initial_capital,
                'max_drawdown': 0
            }
            return summary, pd.DataFrame()  # Return empty DataFrame
        
        df_trades = pd.DataFrame(self.trades)
        
        winning_trades = df_trades[df_trades['pct_return'] > 0]
        losing_trades = df_trades[df_trades['pct_return'] <= 0]
        
        summary = {
            'total_trades': len(df_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(df_trades) * 100 if len(df_trades) > 0 else 0,
            'avg_return_pct': df_trades['pct_return'].mean(),
            'avg_winning_return': winning_trades['pct_return'].mean() if len(winning_trades) > 0 else 0,
            'avg_losing_return': losing_trades['pct_return'].mean() if len(losing_trades) > 0 else 0,
            'total_return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'final_capital': self.capital,
            'max_drawdown': self.calculate_max_drawdown(df_trades)
        }
        
        return summary, df_trades
        
        df_trades = pd.DataFrame(self.trades)
        
        winning_trades = df_trades[df_trades['pct_return'] > 0]
        losing_trades = df_trades[df_trades['pct_return'] <= 0]
        
        summary = {
            'total_trades': len(df_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(df_trades) * 100 if len(df_trades) > 0 else 0,
            'avg_return_pct': df_trades['pct_return'].mean(),
            'avg_winning_return': winning_trades['pct_return'].mean() if len(winning_trades) > 0 else 0,
            'avg_losing_return': losing_trades['pct_return'].mean() if len(losing_trades) > 0 else 0,
            'total_return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'final_capital': self.capital,
            'max_drawdown': self.calculate_max_drawdown(df_trades)
        }
        
        return summary, df_trades
    
    def calculate_max_drawdown(self, df_trades):
        """Calculate maximum drawdown from trade history."""
        if df_trades.empty:
            return 0
        
        cumulative = [self.initial_capital]
        cumulative.extend(df_trades['capital_after'].tolist())
        cumulative = np.array(cumulative)
        
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        
        return drawdown.min()

if __name__ == "__main__":
    data_path = "/Users/admin/FinAi/market_data/"
    tickers = get_symbols_from_folder(data_path)
    
    tickers = filter_sp500_tickers(tickers)
        
    # Load model
    model_path = "/Users/admin/FinAi/"
    model = daily_check.load_model(model_path)
    #tickers = ["UNP", "UPS", "COP", "MTCH", "DVN", "MGM", "MOS", "GPC", "DVA"]
    
    # Portfolio tracking
    initial_capital = 10000
    portfolio_results = {}
    
    ticker_gains_map = np.load('/Users/admin/FinAi/ticker_gains_map.npy', allow_pickle=True).item()
    
    for ticker in tickers:
        
        # Skip if ticker not in map
        if ticker not in ticker_gains_map:
            print(f"Skipping {ticker} - not in gains map")
            continue
        
        # Skip if gains are ≤20%
        if not ticker_gains_map[ticker]:
            continue
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}\n")

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

        # Align prices with tr_labels
        window_days = 60
        aligned_prices = df["Close"].iloc[-len(tr_labels):]
        if len(aligned_prices) != len(tr_labels):
            print(f"[Error] Label-price mismatch for {ticker}")
            continue
        
        print(f"Data points: {len(tr_labels)}")
        print(f"Date range: {aligned_prices.index[0]} to {aligned_prices.index[-1]}")

        # Predict
        pred_test = model.predict(tr_data)
        assert pred_test.shape[0] == len(aligned_prices), "Prediction count mismatch with prices"
        
        prices_np = aligned_prices.to_numpy().ravel()
        confidences = np.max(pred_test, axis=1)
        pred_classes = np.argmax(pred_test, axis=1)
        
        doConfidencePlot = False
        if doConfidencePlot:
            # Assuming you already have: prices_np, pred_test (shape: [n, n_classes]), pred_classes
            x = np.arange(len(prices_np))
            n_classes = pred_test.shape[1]
            
            # --- Smooth confidences ---
            window = 2  # adjust for smoother / more responsive
            smoothed_conf = pd.DataFrame(pred_test).rolling(window, min_periods=1).mean().to_numpy()
            
            # --- Plot ---
            fig, ax1 = plt.subplots(figsize=(14, 6))
            
            # Price line
            ax1.plot(x, prices_np, color='black', linewidth=1.5, label='Price')
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price", color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            
            # Secondary axis for confidences
            ax2 = ax1.twinx()
            colors = ['tab:blue', 'tab:orange', 'tab:green']
            
            for i in range(n_classes):
                ax2.plot(
                    x, smoothed_conf[:, i],
                    color=colors[i % len(colors)],
                    label=f"Class {i} smoothed conf (w={window})",
                    alpha=0.8
                )
            
            ax2.set_ylabel("Smoothed Confidence", color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            ax2.set_ylim(0, 1.0)
            
            # Optional scatter for buy/sell classes
            buy_idx = np.where(pred_classes == 2)[0]
            sell_idx = np.where(pred_classes == 1)[0]
            ax1.scatter(buy_idx, prices_np[buy_idx], color='green', label='BUY (class 2)', s=30, marker='^')
            ax1.scatter(sell_idx, prices_np[sell_idx], color='red', label='SELL (class 1)', s=30, marker='v')
            
            # Combined legend
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title(f"Price vs Smoothed Class Confidences ({n_classes} classes, window={window})")
            plt.tight_layout()
            plt.show()

        # Basic: use raw confidences, 3-day rising window, default classes (buy_class=2,sell_class=1)
        res = directional_confidence_signals(
            pred_test,
            trend_window=3,
            conf_th=0.8,
        )

        # Apply price filters
        buy_mask = res['buy_mask'].copy()
        sell_mask = res['sell_mask'].copy()

        # 4. Use filtered masks
        buy_pred_idxs = np.where(buy_mask)[0]
        sell_pred_idxs = np.where(sell_mask)[0]
        
        # --- SIMULATE TRADES ---
        simulator = TradeSimulator(initial_capital=initial_capital)
        
        for i in range(len(prices_np)):
            # Check if we should exit an open position
            simulator.check_exit(i, prices_np[i], aligned_prices.index)
            
            # Check for new entry signals
            if buy_mask[i]:
                simulator.execute_trade(i, 'buy', prices_np[i], aligned_prices.index)
                if simulator.open_position:
                    simulator.open_position['ticker'] = ticker
                    
            elif sell_mask[i]:
                simulator.execute_trade(i, 'sell', prices_np[i], aligned_prices.index)
                if simulator.open_position:
                    simulator.open_position['ticker'] = ticker
        
        # Close any remaining position
        simulator.close_final_position(prices_np[-1], aligned_prices.index[-1])
        
        # Get results
        summary, trades_df = simulator.get_performance_summary()
        portfolio_results[ticker] = summary
        
        # Print summary
        print(f"\n📊 Performance Summary for {ticker}:")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.2f}%")
        print(f"   Avg Return per Trade: {summary['avg_return_pct']:.2f}%")
        print(f"   Total Return: {summary['total_return_pct']:.2f}%")
        print(f"   Final Capital: ${summary['final_capital']:.2f}")
        print(f"   Max Drawdown: {summary['max_drawdown']:.2f}%")
        
        if summary['total_trades'] > 0:
            print(f"\n📋 Trade Details:")
            print(trades_df[['entry_date', 'exit_date', 'type', 'entry_price', 
                            'exit_price', 'pct_return', 'days_held', 'exit_reason', 
                            'capital_after']].to_string())
        
        doTradingPlots = True
        if doTradingPlots:
            # Plot
            plt.figure(figsize=(14, 8))
            
            # Price chart
            plt.subplot(2, 1, 1)
            plt.plot(aligned_prices, label='Price', linewidth=1.5)
            plt.plot(aligned_prices.index[buy_pred_idxs], aligned_prices.iloc[buy_pred_idxs],
                     'g^', markersize=10, label='Buy Signal', alpha=0.7)
            plt.plot(aligned_prices.index[sell_pred_idxs], aligned_prices.iloc[sell_pred_idxs],
                     'rv', markersize=10, label='Sell Signal', alpha=0.7)
            plt.title(f'{ticker} - Signals and Price Action')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Equity curve (step function showing flat capital when no position)
            plt.subplot(2, 1, 2)
            if summary['total_trades'] > 0:
                # Create step function: capital stays flat until trade closes
                equity_values = []
                equity_dates = []
                
                current_capital = initial_capital
                equity_values.append(current_capital)
                equity_dates.append(aligned_prices.index[0])
                
                for _, trade in trades_df.iterrows():
                    # Capital stays flat from previous close to new trade entry
                    equity_values.append(current_capital)
                    equity_dates.append(trade['entry_date'])
                    
                    # Capital changes at trade exit
                    current_capital = trade['capital_after']
                    equity_values.append(current_capital)
                    equity_dates.append(trade['exit_date'])
                
                # Extend to end of data
                equity_values.append(current_capital)
                equity_dates.append(aligned_prices.index[-1])
                
                plt.plot(equity_dates, equity_values, 'b-', linewidth=2, label='Portfolio Value', drawstyle='steps-post')
                plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
                
                # Shade profit/loss regions
                equity_array = np.array(equity_values)
                plt.fill_between(equity_dates, initial_capital, equity_values, 
                               where=equity_array >= initial_capital, 
                               color='green', alpha=0.3, label='Profit', step='post')
                plt.fill_between(equity_dates, initial_capital, equity_values, 
                               where=equity_array < initial_capital, 
                               color='red', alpha=0.3, label='Loss', step='post')
            else:
                # No trades executed
                plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='No Trades')
                
            plt.title(f'{ticker} - Equity Curve')
            plt.ylabel('Portfolio Value ($)')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    
    # Portfolio summary
    print(f"\n{'='*60}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*60}\n")
    
    portfolio_df = pd.DataFrame(portfolio_results).T
    print(portfolio_df[['total_trades', 'win_rate', 'total_return_pct', 
                        'final_capital', 'max_drawdown']].to_string())
    
    # Calculate overall performance correctly
    print(f"\n{'='*60}")
    print("OVERALL PORTFOLIO PERFORMANCE")
    print(f"{'='*60}\n")
    
    # Use the actual number of tickers PROCESSED, not total in folder
    num_tickers_processed = len(portfolio_results)
    
    # Method 1: Average returns across tickers (equal weight)
    avg_return = portfolio_df['total_return_pct'].mean()
    print(f"📊 Method 1 - Average Return Across Tickers:")
    print(f"   Mean Return: {avg_return:.2f}%")
    print(f"   If you invested ${initial_capital:,.2f} in each ticker separately:")
    print(f"   Average Final Capital: ${initial_capital * (1 + avg_return/100):,.2f}")
    
    # Method 2: Parallel allocation (split capital equally)
    capital_per_ticker = initial_capital / num_tickers_processed
    total_final_parallel = portfolio_df['final_capital'].sum() * (capital_per_ticker / initial_capital)
    parallel_return = (total_final_parallel - initial_capital) / initial_capital * 100
    print(f"\n💼 Method 2 - Parallel Allocation (${initial_capital:,.2f} split across {num_tickers_processed} tickers):")
    print(f"   Capital per ticker: ${capital_per_ticker:,.2f}")
    print(f"   Total final capital: ${total_final_parallel:,.2f}")
    print(f"   Overall return: {parallel_return:.2f}%")
    
    # Method 3: Sequential trading (compound across tickers)
    sequential_capital = initial_capital
    print(f"\n🔄 Method 3 - Sequential Trading (trade one after another with ${initial_capital:,.2f}):")
    for ticker in portfolio_results.keys():
        ticker_return = portfolio_results[ticker]['total_return_pct']
        sequential_capital *= (1 + ticker_return / 100)
        print(f"   After {ticker}: ${sequential_capital:,.2f} ({ticker_return:+.2f}%)")
    
    sequential_return = (sequential_capital - initial_capital) / initial_capital * 100
    print(f"\n   Final Capital: ${sequential_capital:,.2f}")
    print(f"   Overall Return: {sequential_return:.2f}%")

    # Create a dictionary map with tickers as keys and boolean as values
    ticker_gains_map = {}
    for ticker, results in portfolio_results.items():
        total_return = results['total_return_pct']
        ticker_gains_map[ticker] = total_return > 20.0
    
    
    # Also create a structured numpy array for more efficient storage
    ticker_gains_structured = np.array(
        [(ticker, gains) for ticker, gains in ticker_gains_map.items()],
        dtype=[('ticker', 'U10'), ('above_20pct', bool)]
    )
    
    saveTickerMap = False
    if saveTickerMap:
        # Save as numpy file (dictionary)
        np.save('ticker_gains_map.npy', ticker_gains_map)
        np.save('ticker_gains_structured.npy', ticker_gains_structured)

    
    print(f"\n{'='*60}")
    print("TICKER GAINS FILTER (>20%)")
    print(f"{'='*60}\n")
    print(f"{'Ticker':<10} {'Return %':<12} {'Above 20%'}")
    print(f"{'-'*40}")
    for ticker, results in portfolio_results.items():
        total_return = results['total_return_pct']
        is_above = ticker_gains_map[ticker]
        print(f"{ticker:<10} {total_return:>10.2f}% {str(is_above):>10}")
    
    print(f"\nSaved files:")
    print(f"  - ticker_gains_map.npy (dictionary)")
    print(f"  - ticker_gains_structured.npy (structured array)")
    
    print(f"\nTickers with >20% gains: {sum(ticker_gains_map.values())}")
    print(f"Tickers with ≤20% gains: {len(ticker_gains_map) - sum(ticker_gains_map.values())}")
    
    print(f"\nTo load:")
    print(f"  map_dict = np.load('ticker_gains_map.npy', allow_pickle=True).item()")
    print(f"  structured = np.load('ticker_gains_structured.npy')")
    print(f"\nAccess examples:")
    print(f"  map_dict['AAPL']  # Returns True/False")
    print(f"  structured['ticker']  # Array of all tickers")
    print(f"  structured['above_20pct']  # Array of all booleans")
    