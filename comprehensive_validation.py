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
        mask = ma_short < ma_long # INVERTED LOGIC OTHERWISE IS TOO LATE
    elif mode == "bear":
        mask = ma_short > ma_long
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


def decluster_signals(signal_mask, min_gap=10, mode="first"):
    """
    Decluster signals by enforcing a minimum gap between them.
    
    Args:
        signal_mask (np.ndarray): Boolean mask of signals.
        min_gap (int): Minimum number of bars between signals.
        mode (str): 
            "first" -> keep the first signal in each cluster
            "last"  -> keep the last signal in each cluster
            "max_conf" -> keep the signal with highest confidence (requires confidences)
    
    Returns:
        np.ndarray: Boolean mask with declustered signals.
    """
    idx = np.where(signal_mask)[0]
    keep = []
    last_kept = -min_gap

    for i in idx:
        if i - last_kept >= min_gap:
            if mode == "first":
                keep.append(i)
                last_kept = i
            elif mode == "last":
                # defer keeping until cluster ends
                if keep and keep[-1] == last_kept:
                    keep.pop()
                keep.append(i)
                last_kept = i
            else:
                raise ValueError("mode must be 'first' or 'last' for now")

    mask = np.zeros_like(signal_mask, dtype=bool)
    mask[keep] = True
    return mask

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
        Check if position should be closed (5 days or 10% gain).
        
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
        
        # Exit conditions: 5 days OR 10% gain
        should_exit = (self.open_position['days_held'] >= 5) or (pct_change >= 0.10)
        
        if should_exit:
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
                'capital_after': exit_value
            }
            
            self.trades.append(trade_record)
            self.capital = exit_value
            self.open_position = None
            return True
        
        return False
    
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
    
    # Load model
    model_path = "/Users/admin/FinAi"
    model = daily_check.load_model(model_path)
    tickers = ["RMS.PA", "LUMN", "RBA", "PRLB", "USPH", "CLF"]
    
    # Portfolio tracking
    initial_capital = 10000
    portfolio_results = {}
    
    for ticker in tickers[0:100]:
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
        
        # Buys
        trend_bull = ma_trend_check(prices_np, short=100, long=200, mode="bull")  # buy zones
        trend_bear = ma_trend_check(prices_np, short=100, long=200, mode="bear")  # sell zones

        buy_raw = pred_classes == 1
        confidence_mask = confidences >= 0.99
        buy_strict = buy_raw & confidence_mask
        minima_mask = strict_rolling_extrema(prices_np, lookback=10, mode="min")
        score = (trend_bull.astype(int) + minima_mask.astype(int))
        buy_mask = (score >= 2) & buy_strict

        # Sells
        sell_raw = pred_classes == 0
        confidence_mask = confidences >= 0.99
        sell_strict = sell_raw & confidence_mask
        maxima_mask = strict_rolling_extrema(prices_np, lookback=10, mode="max")
        score = (trend_bear.astype(int) + maxima_mask.astype(int))
        sell_mask = (score >= 2) & sell_strict
        
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
                            'exit_price', 'pct_return', 'capital_after']].to_string())

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