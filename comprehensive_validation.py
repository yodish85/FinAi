#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected backtesting script with proper timing and no lookahead bias.

Key fixes:
- Signal at day i close -> Execute at day i+1 open (for both entry and exit)
- Proper spread modeling (eToro 0.09%)
- Unit tests to verify timing
"""

import os
import importlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import warnings
import time
import requests
from pathlib import Path

# Import your existing modules (adjust paths as needed)
import extract_features_with_fft
importlib.reload(extract_features_with_fft)

import training_model
importlib.reload(training_model)

import daily_check
importlib.reload(daily_check)

import confirm_trend
importlib.reload(confirm_trend)

from StockFetcher import StockFetcher
from training_model import get_symbols_from_folder

# =====================================================================
# TRADE SIMULATOR - CORRECTED VERSION
# =====================================================================

class TradeSimulator:
    """
    Simulates trades with realistic timing: signals at close, execution at next open.
    - Entry: Signal at day i close → Execute at day i+1 open
    - Exit: Exit condition met at day i close → Execute at day i+1 open
    - Includes eToro-style spreads on both entry and exit
    """
    
    ETORO_SPREADS = {
        'stocks': 0.09,
        'indices': 0.75,
        'commodities': 0.05,
        'crypto': 0.75,
        'forex': 0.0001,
    }
    
    def __init__(
    self,
    initial_capital=10000,
    spread_pct=0.09,
    hold_days=10,
    take_profit_pct=0.10,
    stop_loss_pct=0.05,
    slippage_pct=0.10,        # NEW: max slippage in %
    slippage_mode="random"   # "random" or "fixed"
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.open_position = None
    
        self.spread_pct = spread_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.slippage_mode = slippage_mode
    
        self.total_spread_cost = 0
        self.total_slippage_cost = 0   # NEW
    
        self.hold_days = hold_days
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

    def apply_slippage(self, price, direction):
        """
        direction: 'buy' or 'sell'
        """
        if self.slippage_mode == "fixed":
            slip = self.slippage_pct
        else:
            slip = np.random.uniform(0, self.slippage_pct)
    
        if direction == 'buy':
            slipped_price = price * (1 + slip)
        else:
            slipped_price = price * (1 - slip)
    
        slippage_cost = abs(slipped_price - price)
        return slipped_price, slippage_cost


    def process_signal(self, idx, action, next_open_price, dates):
        """Process a trading signal - execution at next day's open."""
        if self.open_position is not None:
            return
            
        if action == 'buy':
            raw_price = next_open_price * (1 + self.spread_pct)
            entry_price_with_spread, slip_cost = self.apply_slippage(raw_price, 'buy')
            spread_cost = next_open_price * self.spread_pct
            shares = self.capital / entry_price_with_spread
            
            self.open_position = {
                'type': 'long',
                'signal_idx': idx,
                'entry_idx': idx + 1,
                'entry_date': dates[idx + 1],
                'entry_price': entry_price_with_spread,
                'market_price': next_open_price,
                'shares': shares,
                'days_held': 0,
                'entry_spread_cost': spread_cost * shares
            }
            self.total_spread_cost += spread_cost * shares
            self.total_slippage_cost += slip_cost * shares
            
        elif action == 'sell':
            raw_price = next_open_price * (1 - self.spread_pct)
            entry_price_with_spread, slip_cost = self.apply_slippage(raw_price, 'sell')
            spread_cost = next_open_price * self.spread_pct
            shares = self.capital / next_open_price
            
            self.open_position = {
                'type': 'short',
                'signal_idx': idx,
                'entry_idx': idx + 1,
                'entry_date': dates[idx + 1],
                'entry_price': entry_price_with_spread,
                'market_price': next_open_price,
                'shares': shares,
                'days_held': 0,
                'entry_spread_cost': spread_cost * shares
            }
            self.total_spread_cost += spread_cost * shares
            self.total_slippage_cost += slip_cost * shares
    
    def check_and_exit(self, current_idx, current_close, next_open_price, dates):
        """Check exit conditions at current close, execute at next open."""
        if self.open_position is None:
            return False
        
        self.open_position['days_held'] = current_idx - self.open_position['entry_idx']
        
        entry_price = self.open_position['entry_price']
        pos_type = self.open_position['type']
        shares = self.open_position['shares']
        
        # Check P&L at current close
        if pos_type == 'long':
            pct_change_at_close = (current_close - entry_price) / entry_price
        else:
            pct_change_at_close = (entry_price - current_close) / entry_price
        
        # Exit conditions
        hit_take_profit = pct_change_at_close >= self.take_profit_pct
        hit_stop_loss = pct_change_at_close <= -self.stop_loss_pct
        hit_time_limit = self.open_position['days_held'] >= self.hold_days
        
        should_exit = hit_take_profit or hit_stop_loss or hit_time_limit
        
        if should_exit:
            # Execute at NEXT day's open
            if pos_type == 'long':
                raw_price = next_open_price * (1 - self.spread_pct)
                exit_price_with_spread, slip_cost = self.apply_slippage(raw_price, 'sell')
                exit_spread_cost = next_open_price * self.spread_pct * shares
            else:
                raw_price = next_open_price * (1 + self.spread_pct)
                exit_price_with_spread, slip_cost = self.apply_slippage(raw_price, 'buy')
                exit_spread_cost = next_open_price * self.spread_pct * shares
            
            self.total_spread_cost += exit_spread_cost
            self.total_slippage_cost += slip_cost * shares
            
            if pos_type == 'long':
                pct_change = (exit_price_with_spread - entry_price) / entry_price
            else:
                pct_change = (entry_price - exit_price_with_spread) / entry_price
            
            exit_value = self.capital * (1 + pct_change)
            profit = exit_value - self.capital
            
            if hit_take_profit:
                exit_reason = 'take_profit'
            elif hit_stop_loss:
                exit_reason = 'stop_loss'
            elif hit_time_limit:
                exit_reason = 'time_limit'
            else:
                exit_reason = 'unknown'
            
            trade_record = {
                'ticker': self.open_position.get('ticker', 'N/A'),
                'type': pos_type,
                'signal_date': dates[self.open_position['signal_idx']],
                'entry_date': self.open_position['entry_date'],
                'entry_price': entry_price,
                'market_entry_price': self.open_position['market_price'],
                'exit_signal_date': dates[current_idx],
                'exit_date': dates[current_idx + 1] if current_idx + 1 < len(dates) else dates[current_idx],
                'exit_price': exit_price_with_spread,
                'market_exit_price': next_open_price,
                'days_held': self.open_position['days_held'],
                'pct_return': pct_change * 100,
                'profit': profit,
                'capital_after': exit_value,
                'entry_spread_cost': self.open_position['entry_spread_cost'],
                'exit_spread_cost': exit_spread_cost,
                'total_spread_cost': self.open_position['entry_spread_cost'] + exit_spread_cost,
                'exit_reason': exit_reason,
                'slippage_cost': slip_cost * shares
            }
            
            self.trades.append(trade_record)
            self.capital = exit_value
            self.open_position = None
            return True
        
        return False
    
    def close_final_position(self, final_close_price, final_date):
        """Force close any remaining position at final close price."""
        if self.open_position is not None:
            entry_price = self.open_position['entry_price']
            pos_type = self.open_position['type']
            shares = self.open_position['shares']
            
            if pos_type == 'long':
                exit_price_with_spread = final_close_price * (1 - self.spread_pct)
                exit_spread_cost = final_close_price * self.spread_pct * shares
            else:
                exit_price_with_spread = final_close_price * (1 + self.spread_pct)
                exit_spread_cost = final_close_price * self.spread_pct * shares
            
            self.total_spread_cost += exit_spread_cost
            
            if pos_type == 'long':
                pct_change = (exit_price_with_spread - entry_price) / entry_price
            else:
                pct_change = (entry_price - exit_price_with_spread) / entry_price
            
            exit_value = self.capital * (1 + pct_change)
            profit = exit_value - self.capital
            
            trade_record = {
                'ticker': self.open_position.get('ticker', 'N/A'),
                'type': pos_type,
                'signal_date': None,
                'entry_date': self.open_position['entry_date'],
                'entry_price': entry_price,
                'market_entry_price': self.open_position['market_price'],
                'exit_signal_date': final_date,
                'exit_date': final_date,
                'exit_price': exit_price_with_spread,
                'market_exit_price': final_close_price,
                'days_held': self.open_position['days_held'],
                'pct_return': pct_change * 100,
                'profit': profit,
                'capital_after': exit_value,
                'entry_spread_cost': self.open_position['entry_spread_cost'],
                'exit_spread_cost': exit_spread_cost,
                'total_spread_cost': self.open_position['entry_spread_cost'] + exit_spread_cost,
                'exit_reason': 'end_of_data'
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
                'max_drawdown': 0,
                'total_spread_cost': 0,
                'spread_pct_of_initial': 0,
                'avg_spread_per_trade': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'total_slippage_cost': float(self.total_slippage_cost)
            }
            return summary, pd.DataFrame()
        
        df_trades = pd.DataFrame(self.trades)
        
        # Ensure all numeric columns are scalars, not arrays
        for col in ['pct_return', 'profit', 'capital_after', 'entry_spread_cost', 
                    'exit_spread_cost', 'total_spread_cost', 'entry_price', 'exit_price']:
            if col in df_trades.columns:
                df_trades[col] = df_trades[col].apply(
                    lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else float(x)
                )
        
        winning_trades = df_trades[df_trades['pct_return'] > 0]
        losing_trades = df_trades[df_trades['pct_return'] <= 0]
        
        total_wins = float(winning_trades['profit'].sum()) if len(winning_trades) > 0 else 0.0
        total_losses = float(abs(losing_trades['profit'].sum())) if len(losing_trades) > 0 else 0.0
        profit_factor = float(total_wins / total_losses) if total_losses > 0 else 0.0
        
        returns = df_trades['pct_return'].values.astype(float) / 100
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252 / self.hold_days))
        else:
            sharpe_ratio = 0.0
        
        summary = {
            'total_trades': int(len(df_trades)),
            'winning_trades': int(len(winning_trades)),
            'losing_trades': int(len(losing_trades)),
            'win_rate': float(len(winning_trades) / len(df_trades) * 100) if len(df_trades) > 0 else 0.0,
            'avg_return_pct': float(df_trades['pct_return'].mean()),
            'avg_winning_return': float(winning_trades['pct_return'].mean()) if len(winning_trades) > 0 else 0.0,
            'avg_losing_return': float(losing_trades['pct_return'].mean()) if len(losing_trades) > 0 else 0.0,
            'total_return_pct': float((self.capital - self.initial_capital) / self.initial_capital * 100),
            'final_capital': float(self.capital),
            'max_drawdown': float(self.calculate_max_drawdown(df_trades)),
            'total_spread_cost': float(self.total_spread_cost),
            'spread_pct_of_initial': float((self.total_spread_cost / self.initial_capital) * 100),
            'avg_spread_per_trade': float(df_trades['total_spread_cost'].mean()),
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        }
        
        return summary, df_trades
    
    def calculate_max_drawdown(self, df_trades):
        """Calculate maximum drawdown from trade history."""
        if df_trades.empty:
            return 0
        
        # Ensure capital_after values are scalars, not arrays
        capital_values = []
        for val in df_trades['capital_after'].values:
            if isinstance(val, (list, np.ndarray)):
                capital_values.append(float(val[0]) if len(val) > 0 else float(val))
            else:
                capital_values.append(float(val))
        
        cumulative = [self.initial_capital] + capital_values
        cumulative = np.array(cumulative, dtype=float)
        
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        
        return float(drawdown.min())


def run_backtest_corrected(ticker, close_prices, open_prices, dates, 
                           buy_mask, sell_mask, initial_capital=10000):
    """Corrected backtest loop with proper timing."""
    simulator = TradeSimulator(
        initial_capital=initial_capital,
        spread_pct=0.09,
        hold_days=10,
        take_profit_pct=0.1,
        stop_loss_pct=0.05
    )
    
    n = len(close_prices)
    
    for i in range(n - 1):
        current_close = close_prices[i]
        next_open = open_prices[i + 1]
        
        # Check exit first
        if simulator.open_position is not None:
            simulator.check_and_exit(i, current_close, next_open, dates)
        
        # Check for new signals
        if simulator.open_position is None:
            if buy_mask[i]:
                simulator.process_signal(i, 'buy', next_open, dates)
                if simulator.open_position:
                    simulator.open_position['ticker'] = ticker
                    
            elif sell_mask[i]:
                simulator.process_signal(i, 'sell', next_open, dates)
                if simulator.open_position:
                    simulator.open_position['ticker'] = ticker
    
    # Close final position
    if simulator.open_position is not None:
        simulator.close_final_position(close_prices[-1], dates[-1])
    
    return simulator


# =====================================================================
# UNIT TESTS
# =====================================================================

def verify_trade_timing(trades_df):
    """Verify that trades follow correct timing rules."""
    if trades_df.empty:
        print("⚠️  No trades to verify")
        return True
    
    all_valid = True
    for idx, trade in trades_df.iterrows():
        signal_date = trade['signal_date']
        entry_date = trade['entry_date']
        exit_signal_date = trade['exit_signal_date']
        exit_date = trade['exit_date']
        
        # Entry should be 1 business day after signal
        if pd.notna(signal_date):
            days_to_entry = np.busday_count(signal_date.date(), entry_date.date())
            if days_to_entry != 1:
                print(f"❌ Trade {idx}: Entry timing wrong (signal: {signal_date}, entry: {entry_date}, business days: {days_to_entry})")
                all_valid = False
        
        # Exit should be 1 business day after exit signal
        if exit_date != exit_signal_date:
            days_to_exit = np.busday_count(exit_signal_date.date(), exit_date.date())
            if days_to_exit != 1:
                print(f"❌ Trade {idx}: Exit timing wrong (exit signal: {exit_signal_date}, exit: {exit_date}, business days: {days_to_exit})")
                all_valid = False
    
    if all_valid:
        print("✅ All trades follow correct timing!")
    return all_valid


def run_unit_tests():
    """Run unit tests on synthetic data."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60 + "\n")
    
    # Create synthetic data
    dates = pd.date_range('2024-01-01', periods=20, freq='B')
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(20) * 2)
    open_prices = close_prices + np.random.randn(20) * 0.5
    
    # Test 1: Single buy signal
    print("Test 1: Single buy signal on day 0")
    buy_mask = np.zeros(20, dtype=bool)
    buy_mask[0] = True
    sell_mask = np.zeros(20, dtype=bool)
    
    sim = run_backtest_corrected('TEST', close_prices, open_prices, dates, buy_mask, sell_mask, 10000)
    summary, trades = sim.get_performance_summary()
    
    if summary['total_trades'] == 1:
        print("✅ Generated 1 trade as expected")
        trade = trades.iloc[0]
        print(f"   Signal date: {trade['signal_date']}")
        print(f"   Entry date: {trade['entry_date']}")
        print(f"   Entry should be 1 day after signal: {(trade['entry_date'] - trade['signal_date']).days} days")
        verify_trade_timing(trades)
    else:
        print(f"❌ Expected 1 trade, got {summary['total_trades']}")
    
    # Test 2: Multiple signals
    print("\nTest 2: Multiple signals with proper spacing")
    buy_mask = np.zeros(20, dtype=bool)
    buy_mask[0] = True
    buy_mask[10] = True
    sell_mask = np.zeros(20, dtype=bool)
    
    sim = run_backtest_corrected('TEST', close_prices, open_prices, dates, buy_mask, sell_mask, 10000)
    summary, trades = sim.get_performance_summary()
    
    print(f"   Total trades: {summary['total_trades']}")
    verify_trade_timing(trades)
    
    # Test 3: Spread cost verification
    print("\nTest 3: Spread cost calculation")
    buy_mask = np.zeros(20, dtype=bool)
    buy_mask[0] = True
    sell_mask = np.zeros(20, dtype=bool)
    
    sim = run_backtest_corrected('TEST', close_prices, open_prices, dates, buy_mask, sell_mask, 10000)
    summary, trades = sim.get_performance_summary()
    
    if summary['total_trades'] > 0:
        trade = trades.iloc[0]
        expected_entry_spread = trade['market_entry_price'] * 0.0009 * (10000 / (trade['market_entry_price'] * 1.0009))
        actual_entry_spread = trade['entry_spread_cost']
        spread_diff = abs(expected_entry_spread - actual_entry_spread)
        
        if spread_diff < 0.01:
            print(f"✅ Spread calculation correct (diff: ${spread_diff:.4f})")
        else:
            print(f"❌ Spread calculation wrong (expected: ${expected_entry_spread:.2f}, got: ${actual_entry_spread:.2f})")
    
    print("\n" + "="*60)
    print("UNIT TESTS COMPLETE")
    print("="*60 + "\n")


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

CACHE_PATH = Path("/Users/admin/FinAi/market_data/sp500_tickers.csv")
CACHE_TTL = 24 * 3600

def fetch_sp500_from_wikipedia():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    tables = pd.read_html(r.text, displayed_only=False)
    for df in tables:
        if 'Symbol' in df.columns:
            return [s.replace('.', '-') for s in df['Symbol'].astype(str).tolist()]
    raise RuntimeError("Could not find Symbol column")

def read_sp500_from_cache():
    if not CACHE_PATH.exists():
        return None
    if time.time() - CACHE_PATH.stat().st_mtime > CACHE_TTL:
        return None
    return pd.read_csv(CACHE_PATH)['symbol'].astype(str).tolist()

def write_sp500_cache(symbols):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(CACHE_PATH, index=False)

def get_sp500_tickers():
    symbols = read_sp500_from_cache()
    if symbols:
        return symbols
    try:
        symbols = fetch_sp500_from_wikipedia()
        write_sp500_cache(symbols)
        return symbols
    except Exception as e:
        print(f"Warning: {e}")
        local = Path("data/sp500_static.csv")
        if local.exists():
            return pd.read_csv(local)['symbol'].astype(str).tolist()
        raise RuntimeError("Could not obtain S&P 500 tickers")

def filter_sp500_tickers(tickers):
    sp500 = set(get_sp500_tickers())
    return [t for t in tickers if t in sp500]

def directional_confidence_signals(pred_test, trend_window=5, conf_th=0.0,
                                   smooth_window=3, window_offset=0):
    """Generate directional signals from predictions."""
    pred_test = np.asarray(pred_test)
    n = pred_test.shape[0]
    c0, c1, c2 = pred_test[:, 0], pred_test[:, 1], pred_test[:, 2]
    
    c0 = pd.DataFrame(c0).rolling(smooth_window, min_periods=1).mean().to_numpy().flatten()
    c1 = pd.DataFrame(c1).rolling(smooth_window, min_periods=1).mean().to_numpy().flatten()
    c2 = pd.DataFrame(c2).rolling(smooth_window, min_periods=1).mean().to_numpy().flatten()
    
    buy_mask = np.zeros(n, dtype=bool)
    sell_mask = np.zeros(n, dtype=bool)
    
    for t in range(n):
        if t < window_offset:
            continue
        end = t - window_offset + 1
        start = max(0, end - trend_window)
        
        c0_win = c0[start:end]
        c1_win = c1[start:end]
        c2_win = c2[start:end]
        c0_t, c1_t, c2_t = c0[t], c1[t], c2[t]
        
        if len(c0_win) == 0:
            continue
        
        if c2_t >= np.max(c2_win) and c1_t <= np.min(c1_win) and c0_t <= np.min(c0_win) and c2_t >= conf_th:
            buy_mask[t] = True
        
        if c1_t >= np.max(c1_win) and c2_t <= np.min(c2_win) and c0_t <= np.min(c0_win) and c1_t >= conf_th:
            sell_mask[t] = True
    
    return {
        "buy_mask": buy_mask,
        "sell_mask": sell_mask,
        "buy_idx": np.where(buy_mask)[0],
        "sell_idx": np.where(sell_mask)[0]
    }


def analyze_trade_exits(trades_df):
    """
    Analyze why trades are exiting and their performance.
    """
    if trades_df.empty:
        print("No trades to analyze")
        return
    
    print("\n" + "="*70)
    print("TRADE EXIT ANALYSIS")
    print("="*70)
    
    # Exit reason breakdown
    exit_reasons = trades_df['exit_reason'].value_counts()
    print("\n📊 Exit Reasons:")
    for reason, count in exit_reasons.items():
        pct = count / len(trades_df) * 100
        print(f"   {reason:15s}: {count:3d} trades ({pct:5.1f}%)")
    
    # Performance by exit reason
    print("\n💰 Average Return by Exit Reason:")
    for reason in exit_reasons.index:
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        avg_return = reason_trades['pct_return'].mean()
        win_rate = (reason_trades['pct_return'] > 0).sum() / len(reason_trades) * 100
        print(f"   {reason:15s}: {avg_return:+7.2f}% avg return, {win_rate:.1f}% win rate")
    
    # Days held analysis
    print("\n⏱️  Days Held Analysis:")
    print(f"   Average: {trades_df['days_held'].mean():.1f} days")
    print(f"   Median:  {trades_df['days_held'].median():.1f} days")
    print(f"   Min/Max: {trades_df['days_held'].min():.0f} / {trades_df['days_held'].max():.0f} days")
    
    # Winners vs losers
    winners = trades_df[trades_df['pct_return'] > 0]
    losers = trades_df[trades_df['pct_return'] <= 0]
    
    print("\n🎯 Winners vs Losers:")
    print(f"   Winners: {len(winners):3d} trades, avg return: {winners['pct_return'].mean():+.2f}%")
    print(f"   Losers:  {len(losers):3d} trades, avg return: {losers['pct_return'].mean():+.2f}%")
    print(f"   Win rate: {len(winners)/len(trades_df)*100:.1f}%")
    
    # The critical insight: What if we held longer?
    time_limit_trades = trades_df[trades_df['exit_reason'] == 'time_limit']
    if len(time_limit_trades) > 0:
        tl_winners = time_limit_trades[time_limit_trades['pct_return'] > 0]
        tl_losers = time_limit_trades[time_limit_trades['pct_return'] <= 0]
        
        print("\n⚠️  TIME LIMIT Trades (hit 5-day hold limit):")
        print(f"   Total: {len(time_limit_trades)}")
        print(f"   Winners: {len(tl_winners)} (avg: {tl_winners['pct_return'].mean():+.2f}%)")
        print(f"   Losers: {len(tl_losers)} (avg: {tl_losers['pct_return'].mean():+.2f}%)")
        
        # How many time-limit losers are small losses?
        small_losers = tl_losers[tl_losers['pct_return'] > -2]  # Less than 2% loss
        if len(small_losers) > 0:
            print(f"\n   ⚡ INSIGHT: {len(small_losers)} time-limit exits with <2% loss")
            print(f"      These might have recovered if held longer!")
    
    # Stop loss analysis
    stop_loss_trades = trades_df[trades_df['exit_reason'] == 'stop_loss']
    if len(stop_loss_trades) > 0:
        print("\n🛑 STOP LOSS Trades:")
        print(f"   Total: {len(stop_loss_trades)}")
        print(f"   Avg loss: {stop_loss_trades['pct_return'].mean():.2f}%")
        print(f"   These are PROTECTING you - they hit the 5% stop")
    
    # Take profit analysis
    take_profit_trades = trades_df[trades_df['exit_reason'] == 'take_profit']
    if len(take_profit_trades) > 0:
        print("\n✅ TAKE PROFIT Trades:")
        print(f"   Total: {len(take_profit_trades)}")
        print(f"   Avg gain: {take_profit_trades['pct_return'].mean():.2f}%")
        print(f"   These hit your 10% target - good trades!")
    
    return {
        'exit_reasons': exit_reasons,
        'winners': winners,
        'losers': losers,
        'time_limit_trades': time_limit_trades,
        'stop_loss_trades': stop_loss_trades,
        'take_profit_trades': take_profit_trades
    }


def suggest_improvements(trades_df):
    """
    Suggest specific improvements based on trade data.
    """
    print("\n" + "="*70)
    print("💡 SUGGESTED IMPROVEMENTS")
    print("="*70)
    
    analysis = analyze_trade_exits(trades_df)
    
    suggestions = []
    
    # Check if too many time limit exits
    time_limit_pct = (trades_df['exit_reason'] == 'time_limit').sum() / len(trades_df) * 100
    if time_limit_pct > 50:
        avg_tl_return = trades_df[trades_df['exit_reason'] == 'time_limit']['pct_return'].mean()
        suggestions.append({
            'priority': 'HIGH',
            'issue': f'{time_limit_pct:.0f}% of trades hit time limit',
            'recommendation': f'Increase hold_days' if avg_tl_return > 0 else 'Keep current hold period'
        })
    
    # Check if stop loss is too tight
    stop_loss_pct = (trades_df['exit_reason'] == 'stop_loss').sum() / len(trades_df) * 100
    if stop_loss_pct > 30:
        suggestions.append({
            'priority': 'HIGH',
            'issue': f'{stop_loss_pct:.0f}% of trades hit stop loss',
            'recommendation': 'Your stop loss might be too tight.'
        })
    
    # Check if take profit is too high
    take_profit_pct = (trades_df['exit_reason'] == 'take_profit').sum() / len(trades_df) * 100
    if take_profit_pct < 10:
        suggestions.append({
            'priority': 'MEDIUM',
            'issue': f'Only {take_profit_pct:.0f}% of trades hit take profit',
            'recommendation': 'Lower take profit to lock in more winners'
        })
    
    # Check win rate
    win_rate = (trades_df['pct_return'] > 0).sum() / len(trades_df) * 100
    if win_rate < 50:
        suggestions.append({
            'priority': 'HIGH',
            'issue': f'Win rate is {win_rate:.1f}% (below 50%)',
            'recommendation': 'Add stricter filters OR widen stops to avoid whipsaws'
        })
    
    # Check profit factor
    total_wins = trades_df[trades_df['pct_return'] > 0]['profit'].sum()
    total_losses = abs(trades_df[trades_df['pct_return'] <= 0]['profit'].sum())
    if total_losses > 0:
        profit_factor = total_wins / total_losses
        if profit_factor < 1.5:
            suggestions.append({
                'priority': 'MEDIUM',
                'issue': f'Profit factor is {profit_factor:.2f} (should be >1.5)',
                'recommendation': 'Either improve win rate OR increase avg win / decrease avg loss'
            })
    
    # Print suggestions
    if suggestions:
        print("\n🔧 Specific Action Items:\n")
        for i, sug in enumerate(suggestions, 1):
            print(f"{i}. [{sug['priority']}] {sug['issue']}")
            print(f"   → {sug['recommendation']}\n")
    else:
        print("\n✅ Your position management looks reasonable!")
    
    return suggestions


def plot_trade_distribution(trades_df):
    """
    Visualize trade return distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Return distribution
    ax1 = axes[0, 0]
    returns = trades_df['pct_return']
    ax1.hist(returns, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.axvline(x=returns.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Number of Trades')
    ax1.set_title('Trade Return Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Exit reasons
    ax2 = axes[0, 1]
    exit_counts = trades_df['exit_reason'].value_counts()
    colors = {'take_profit': 'green', 'stop_loss': 'red', 'time_limit': 'orange', 'end_of_data': 'gray'}
    bar_colors = [colors.get(reason, 'blue') for reason in exit_counts.index]
    ax2.bar(range(len(exit_counts)), exit_counts.values, color=bar_colors, edgecolor='black')
    ax2.set_xticks(range(len(exit_counts)))
    ax2.set_xticklabels(exit_counts.index, rotation=45, ha='right')
    ax2.set_ylabel('Number of Trades')
    ax2.set_title('Exit Reasons')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Returns by exit reason
    ax3 = axes[1, 0]
    exit_reasons = trades_df['exit_reason'].unique()
    avg_returns = [trades_df[trades_df['exit_reason'] == reason]['pct_return'].mean() 
                   for reason in exit_reasons]
    bar_colors = ['green' if r > 0 else 'red' for r in avg_returns]
    ax3.bar(range(len(exit_reasons)), avg_returns, color=bar_colors, edgecolor='black', alpha=0.7)
    ax3.set_xticks(range(len(exit_reasons)))
    ax3.set_xticklabels(exit_reasons, rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Average Return (%)')
    ax3.set_title('Average Return by Exit Reason')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Days held vs return
    ax4 = axes[1, 1]
    colors = ['green' if r > 0 else 'red' for r in trades_df['pct_return']]
    ax4.scatter(trades_df['days_held'], trades_df['pct_return'], c=colors, alpha=0.6, edgecolors='black')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Days Held')
    ax4.set_ylabel('Return (%)')
    ax4.set_title('Days Held vs Return')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


# =====================================================================
# HOW TO USE IN YOUR BACKTEST
# =====================================================================

"""
ADD THIS TO YOUR MAIN BACKTEST LOOP, AFTER GETTING trades_df:

    # Get results
    summary, trades_df = simulator.get_performance_summary()
    
    # ADD THESE LINES:
    analysis_results = analyze_trade_exits(trades_df)
    suggestions = suggest_improvements(trades_df)
    
    # Optionally plot
    if len(trades_df) > 10:
        fig = plot_trade_distribution(trades_df)
        plt.show()
    
    # Continue with your existing plotting...

This will tell you EXACTLY what's wrong with your position management.
"""
# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    
    # Run unit tests first
    run_unit_tests()
    
    # Configuration
    data_path = "/Users/admin/FinAi/market_data/"
    model_path = "/Users/admin/FinAi/"
    initial_capital = 10000
    
    # Load tickers
    tickers = get_symbols_from_folder(data_path)
    tickers = filter_sp500_tickers(tickers)
    
    # Load model
    model = daily_check.load_model(model_path)
    
    # Test on specific tickers
    #tickers = ["TROW", "HPQ", "SMCI", "PYPL", "TPL", "DVA"]
    
    portfolio_results = {}
    
    for ticker in tickers[0:10]:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}\n")
        
        # Fetch data
        print("🔄 Fetching fresh data...")
        fetcher = StockFetcher(base_path=data_path)
        fetcher.fetch_and_save([ticker], data_path)
        
        days_to_process = 1001
        result = extract_features_with_fft.extract_features_with_fft(
            [ticker], data_path, True, 'daily', days_to_process, False
        )
        
        if result is None:
            print(f"[Warning] Skipping {ticker}")
            continue
        
        tr_data, tr_labels, tr_symbols = result
        
        # Get price data
        df = yf.download(ticker, start='2015-01-01', progress=False)
        if df.empty:
            print(f"[Warning] No price data for {ticker}")
            continue
        
        aligned_close = df["Close"].iloc[-len(tr_labels):]
        aligned_open = df["Open"].iloc[-len(tr_labels):]
        
        if len(aligned_close) != len(tr_labels):
            print(f"[Error] Data mismatch for {ticker}")
            continue
        
        print(f"Data points: {len(tr_labels)}")
        print(f"Date range: {aligned_close.index[0]} to {aligned_close.index[-1]}")
        
        # Predict
        pred_test = model.predict(tr_data, verbose=0)
        
        # Generate signals
        conf_th = 0.7
        res = directional_confidence_signals(pred_test, trend_window=3, conf_th=conf_th)
        
        aligned_close = df["Close"].iloc[-len(tr_labels):]
        aligned_open = df["Open"].iloc[-len(tr_labels):]
        aligned_volume = df["Volume"].iloc[-len(tr_labels):]  # ADD THIS
        
        # APPLY FILTERS HERE
        filter_results = confirm_trend.apply_trend_filters(
            buy_mask=res['buy_mask'],
            sell_mask=res['sell_mask'],
            close_prices=aligned_close.to_numpy(),
            volumes=aligned_volume.to_numpy(),
            **confirm_trend.get_filter_preset('aggressive')
        )
        
        # Use filtered signals
        #buy_mask = filter_results['buy_mask']
        #sell_mask = filter_results['sell_mask']
        
        buy_mask = res['buy_mask']
        sell_mask = res['sell_mask']
        
        # Run backtest with corrected simulator
        simulator = run_backtest_corrected(
            ticker=ticker,
            close_prices=aligned_close.to_numpy(),
            open_prices=aligned_open.to_numpy(),
            dates=aligned_close.index,
            buy_mask=buy_mask,
            sell_mask=sell_mask,
            initial_capital=initial_capital
        )
        
        # Get results
        summary, trades_df = simulator.get_performance_summary()
        '''
        analysis_results = analyze_trade_exits(trades_df)
        suggestions = suggest_improvements(trades_df)
        
        # Optionally plot
        if len(trades_df) > 1:
            fig = plot_trade_distribution(trades_df)
            plt.show()
        '''    
        portfolio_results[ticker] = summary
        
        # Verify timing
        verify_trade_timing(trades_df)
        
        # Print summary
        print(f"\n📊 Performance Summary for {ticker}:")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.2f}%")
        print(f"   Avg Return per Trade: {summary['avg_return_pct']:.2f}%")
        print(f"   Total Return: {summary['total_return_pct']:.2f}%")
        print(f"   Final Capital: ${summary['final_capital']:.2f}")
        print(f"   Max Drawdown: {summary['max_drawdown']:.2f}%")
        print(f"   Profit Factor: {summary['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        
        if summary['total_trades'] > 0:
            print(f"\n📋 Trade Details:")
            print(trades_df[['signal_date', 'entry_date', 'exit_date', 'type', 
                            'entry_price', 'exit_price', 'pct_return', 'days_held', 
                            'exit_reason', 'capital_after']].to_string())
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(aligned_close, label='Price', linewidth=1.5)
        buy_idx = res['buy_idx']
        sell_idx = res['sell_idx']
        plt.plot(aligned_close.index[buy_idx], aligned_close.iloc[buy_idx],
                 'g^', markersize=10, label='Buy Signal', alpha=0.7)
        plt.plot(aligned_close.index[sell_idx], aligned_close.iloc[sell_idx],
                 'rv', markersize=10, label='Sell Signal', alpha=0.7)
        plt.title(f'{ticker} - Signals and Price Action')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 1, 2)
        if summary['total_trades'] > 0:
            equity_values = [initial_capital]
            equity_dates = [aligned_close.index[0]]
            current_capital = initial_capital
            
            for _, trade in trades_df.iterrows():
                equity_values.append(current_capital)
                equity_dates.append(trade['entry_date'])
                current_capital = trade['capital_after']
                equity_values.append(current_capital)
                equity_dates.append(trade['exit_date'])
            
            equity_values.append(current_capital)
            equity_dates.append(aligned_close.index[-1])
            
            plt.plot(equity_dates, equity_values, 'b-', linewidth=2, label='Portfolio Value', drawstyle='steps-post')
            plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
            
            equity_array = np.array(equity_values)
            plt.fill_between(equity_dates, initial_capital, equity_values, 
                           where=equity_array >= initial_capital, 
                           color='green', alpha=0.3, label='Profit', step='post')
            plt.fill_between(equity_dates, initial_capital, equity_values, 
                           where=equity_array < initial_capital, 
                           color='red', alpha=0.3, label='Loss', step='post')
        else:
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
    
    # Overall performance
    print(f"\n{'='*60}")
    print("OVERALL PORTFOLIO PERFORMANCE")
    print(f"{'='*60}\n")
    
    num_tickers_processed = len(portfolio_results)
    
    # Method 1: Average returns
    avg_return = portfolio_df['total_return_pct'].mean()
    print(f"📊 Method 1 - Average Return Across Tickers:")
    print(f"   Mean Return: {avg_return:.2f}%")
    print(f"   Average Final Capital: ${initial_capital * (1 + avg_return/100):,.2f}")
    
    # Method 2: Parallel allocation
    capital_per_ticker = initial_capital / num_tickers_processed
    total_final_parallel = portfolio_df['final_capital'].sum() * (capital_per_ticker / initial_capital)
    parallel_return = (total_final_parallel - initial_capital) / initial_capital * 100
    print(f"\n💼 Method 2 - Parallel Allocation:")
    print(f"   Capital per ticker: ${capital_per_ticker:,.2f}")
    print(f"   Total final capital: ${total_final_parallel:,.2f}")
    print(f"   Overall return: {parallel_return:.2f}%")
    
    # Method 3: Sequential
    sequential_capital = initial_capital
    print(f"\n🔄 Method 3 - Sequential Trading:")
    for ticker in portfolio_results.keys():
        ticker_return = portfolio_results[ticker]['total_return_pct']
        sequential_capital *= (1 + ticker_return / 100)
        print(f"   After {ticker}: ${sequential_capital:,.2f} ({ticker_return:+.2f}%)")
    
    sequential_return = (sequential_capital - initial_capital) / initial_capital * 100
    print(f"\n   Final Capital: ${sequential_capital:,.2f}")
    print(f"   Overall Return: {sequential_return:.2f}%")