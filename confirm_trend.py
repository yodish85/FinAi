#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADD THIS CODE TO YOUR EXISTING BACKTEST SCRIPT

Insert after the directional_confidence_signals() call in main loop.
This applies MA, Volume, and RSI filters to your signals.
"""

import numpy as np
import pandas as pd


# =====================================================================
# TECHNICAL INDICATOR CALCULATIONS
# =====================================================================

def calculate_sma(prices, period):
    """Calculate Simple Moving Average."""
    # Flatten to 1D if needed
    if isinstance(prices, np.ndarray) and prices.ndim > 1:
        prices = prices.flatten()
    return pd.Series(prices).rolling(window=period, min_periods=1).mean().values


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    # Flatten to 1D if needed
    if isinstance(prices, np.ndarray) and prices.ndim > 1:
        prices = prices.flatten()
    prices = pd.Series(prices)
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).values


def calculate_volume_sma(volumes, period=20):
    """Calculate volume moving average."""
    # Flatten to 1D if needed
    if isinstance(volumes, np.ndarray) and volumes.ndim > 1:
        volumes = volumes.flatten()
    return pd.Series(volumes).rolling(window=period, min_periods=1).mean().values


# =====================================================================
# FILTER FUNCTION - CALL THIS IN YOUR MAIN LOOP
# =====================================================================

def apply_trend_filters(
    buy_mask,
    sell_mask,
    close_prices,
    volumes=None,
    # Filter toggles
    use_ma_filter=True,
    ma_period=50,
    use_long_ma_filter=True,
    long_ma_period=200,
    use_volume_filter=True,
    volume_period=20,
    volume_multiplier=1.0,
    use_rsi_filter=True,
    rsi_period=14,
    rsi_buy_max=70,
    rsi_sell_min=30,
    verbose=True
):
    """
    Apply trend filters to buy/sell signals.
    
    Returns:
        dict with:
        - 'buy_mask': filtered buy signals
        - 'sell_mask': filtered sell signals
        - 'buy_idx': indices of buy signals
        - 'sell_idx': indices of sell signals
        - 'stats': filtering statistics
        - 'indicators': calculated indicators (for plotting)
    """
    n = len(buy_mask)
    
    # Copy original masks
    filtered_buy = buy_mask.copy()
    filtered_sell = sell_mask.copy()
    
    # Track statistics
    stats = {
        'signals_before': int(buy_mask.sum() + sell_mask.sum()),
        'buy_before': int(buy_mask.sum()),
        'sell_before': int(sell_mask.sum()),
        'filtered_by_ma': 0,
        'filtered_by_long_ma': 0,
        'filtered_by_volume': 0,
        'filtered_by_rsi': 0
    }
    
    # Calculate indicators
    indicators = {}
    
    if use_ma_filter:
        sma = calculate_sma(close_prices, ma_period)
        indicators['sma'] = sma
    
    if use_long_ma_filter:
        long_sma = calculate_sma(close_prices, long_ma_period)
        indicators['long_sma'] = long_sma
    
    if use_volume_filter and volumes is not None:
        volume_sma = calculate_volume_sma(volumes, volume_period)
        indicators['volume_sma'] = volume_sma
    
    if use_rsi_filter:
        rsi = calculate_rsi(close_prices, rsi_period)
        indicators['rsi'] = rsi
    
    # Apply filters point by point (only if at least one filter is enabled)
    if use_ma_filter or use_long_ma_filter or use_volume_filter or use_rsi_filter:
        for i in range(n):
            # MA Filter: Only buy above MA, only sell below MA
            if use_ma_filter and 'sma' in indicators:
                if filtered_buy[i] and close_prices[i] < sma[i]:
                    filtered_buy[i] = False
                    stats['filtered_by_ma'] += 1
                
                if filtered_sell[i] and close_prices[i] > sma[i]:
                    filtered_sell[i] = False
                    stats['filtered_by_ma'] += 1
            
            # Long MA Filter (stronger trend confirmation)
            if use_long_ma_filter and 'long_sma' in indicators:
                if filtered_buy[i] and close_prices[i] < long_sma[i]:
                    filtered_buy[i] = False
                    stats['filtered_by_long_ma'] += 1
                
                if filtered_sell[i] and close_prices[i] > long_sma[i]:
                    filtered_sell[i] = False
                    stats['filtered_by_long_ma'] += 1
            
            # Volume Filter: Require volume above average
            if use_volume_filter and volumes is not None and 'volume_sma' in indicators:
                threshold = volume_sma[i] * volume_multiplier
                if (filtered_buy[i] or filtered_sell[i]) and volumes[i] < threshold:
                    if filtered_buy[i]:
                        filtered_buy[i] = False
                        stats['filtered_by_volume'] += 1
                    if filtered_sell[i]:
                        filtered_sell[i] = False
                        stats['filtered_by_volume'] += 1
            
            # RSI Filter: Avoid overbought buys and oversold sells
            if use_rsi_filter and 'rsi' in indicators:
                if filtered_buy[i] and rsi[i] > rsi_buy_max:
                    filtered_buy[i] = False
                    stats['filtered_by_rsi'] += 1
                
                if filtered_sell[i] and rsi[i] < rsi_sell_min:
                    filtered_sell[i] = False
                    stats['filtered_by_rsi'] += 1
    
    # Update final statistics
    stats['signals_after'] = int(filtered_buy.sum() + filtered_sell.sum())
    stats['buy_after'] = int(filtered_buy.sum())
    stats['sell_after'] = int(filtered_sell.sum())
    stats['total_filtered'] = stats['signals_before'] - stats['signals_after']
    
    if verbose:
        print(f"\n🔍 Trend Filter Results:")
        print(f"   Signals before: {stats['signals_before']} (Buy: {stats['buy_before']}, Sell: {stats['sell_before']})")
        print(f"   Signals after:  {stats['signals_after']} (Buy: {stats['buy_after']}, Sell: {stats['sell_after']})")
        print(f"   Filtered out:   {stats['total_filtered']}")
        
        if stats['signals_before'] > 0:
            reduction_pct = (stats['total_filtered'] / stats['signals_before']) * 100
            print(f"   Reduction:      {reduction_pct:.1f}%")
        
        if stats['total_filtered'] > 0:
            print(f"\n   Breakdown:")
            if stats['filtered_by_ma'] > 0:
                print(f"   - MA filter ({ma_period}d):     {stats['filtered_by_ma']}")
            if stats['filtered_by_long_ma'] > 0:
                print(f"   - Long MA ({long_ma_period}d):  {stats['filtered_by_long_ma']}")
            if stats['filtered_by_volume'] > 0:
                print(f"   - Volume filter:  {stats['filtered_by_volume']}")
            if stats['filtered_by_rsi'] > 0:
                print(f"   - RSI filter:     {stats['filtered_by_rsi']}")
    
    return {
        'buy_mask': filtered_buy,
        'sell_mask': filtered_sell,
        'buy_idx': np.where(filtered_buy)[0],
        'sell_idx': np.where(filtered_sell)[0],
        'stats': stats,
        'indicators': indicators
    }


# =====================================================================
# FILTER PRESETS (OPTIONAL)
# =====================================================================

FILTER_PRESETS = {
    'none': {
        'use_ma_filter': False,
        'use_long_ma_filter': False,
        'use_volume_filter': False,
        'use_rsi_filter': False
    },
    'conservative': {
        'use_ma_filter': True,
        'ma_period': 50,
        'use_long_ma_filter': True,
        'long_ma_period': 200,
        'use_volume_filter': True,
        'volume_multiplier': 1.2,
        'use_rsi_filter': True,
        'rsi_buy_max': 65,
        'rsi_sell_min': 35
    },
    'moderate': {
        'use_ma_filter': True,
        'ma_period': 50,
        'use_long_ma_filter': False,
        'use_volume_filter': True,
        'volume_multiplier': 1.0,
        'use_rsi_filter': True,
        'rsi_buy_max': 70,
        'rsi_sell_min': 30
    },
    'aggressive': {
        'use_ma_filter': True,
        'ma_period': 20,
        'use_long_ma_filter': False,
        'use_volume_filter': False,
        'use_rsi_filter': True,
        'rsi_buy_max': 75,
        'rsi_sell_min': 25
    },
    'ma_only': {
        'use_ma_filter': True,
        'ma_period': 50,
        'use_long_ma_filter': False,
        'use_volume_filter': False,
        'use_rsi_filter': False
    }
}


def get_filter_preset(preset_name='moderate'):
    """Get filter configuration from preset."""
    if preset_name not in FILTER_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(FILTER_PRESETS.keys())}")
    return FILTER_PRESETS[preset_name]


# =====================================================================
# HOW TO INTEGRATE INTO YOUR EXISTING CODE
# =====================================================================

"""
STEP-BY-STEP INTEGRATION:

1. Copy all the code above into your backtest script (before if __name__ == "__main__")

2. In your main loop, find this section:

    # Get price data
    df = yf.download(ticker, start='2015-01-01', progress=False)
    aligned_close = df["Close"].iloc[-len(tr_labels):]
    aligned_open = df["Open"].iloc[-len(tr_labels):]
    
    # ADD THIS LINE:
    aligned_volume = df["Volume"].iloc[-len(tr_labels):]

3. Find this section:

    # Generate signals
    conf_th = 0.5
    res = directional_confidence_signals(pred_test, trend_window=3, conf_th=conf_th)
    buy_mask = res['buy_mask']
    sell_mask = res['sell_mask']
    
    # ADD THIS CODE RIGHT AFTER:
    
    # Apply trend filters
    filter_config = get_filter_preset('moderate')  # or use custom config
    filter_results = apply_trend_filters(
        buy_mask=buy_mask,
        sell_mask=sell_mask,
        close_prices=aligned_close.to_numpy(),
        volumes=aligned_volume.to_numpy(),
        **filter_config
    )
    
    # Use filtered signals
    buy_mask = filter_results['buy_mask']
    sell_mask = filter_results['sell_mask']

4. The rest of your code stays exactly the same! The filtered signals will be used automatically.

5. (Optional) To compare filtered vs unfiltered:

    # Save original signals
    original_buy = res['buy_mask'].copy()
    original_sell = res['sell_mask'].copy()
    
    # Apply filters
    filter_results = apply_trend_filters(...)
    filtered_buy = filter_results['buy_mask']
    filtered_sell = filter_results['sell_mask']
    
    # Run backtest twice
    sim_original = run_backtest_corrected(..., buy_mask=original_buy, sell_mask=original_sell)
    sim_filtered = run_backtest_corrected(..., buy_mask=filtered_buy, sell_mask=filtered_sell)
    
    # Compare
    summary_orig, _ = sim_original.get_performance_summary()
    summary_filt, _ = sim_filtered.get_performance_summary()
    
    print(f"\n📊 COMPARISON - {ticker}:")
    print(f"Original  - Trades: {summary_orig['total_trades']:3d}, Return: {summary_orig['total_return_pct']:+7.2f}%")
    print(f"Filtered  - Trades: {summary_filt['total_trades']:3d}, Return: {summary_filt['total_return_pct']:+7.2f}%")

TESTING DIFFERENT FILTERS:

# Try no filters (baseline)
filter_results = apply_trend_filters(buy_mask, sell_mask, close_prices, volumes, **get_filter_preset('none'))

# Try conservative (all filters, strict thresholds)
filter_results = apply_trend_filters(buy_mask, sell_mask, close_prices, volumes, **get_filter_preset('conservative'))

# Try custom settings
filter_results = apply_trend_filters(
    buy_mask, sell_mask, close_prices, volumes,
    use_ma_filter=True,
    ma_period=50,
    use_volume_filter=True,
    volume_multiplier=1.5,  # Require 50% above average volume
    use_rsi_filter=True,
    rsi_buy_max=65,  # Stricter overbought threshold
    rsi_sell_min=35
)

COMPLETE INTEGRATION EXAMPLE:

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}\n")
        
        # ... your existing data loading code ...
        
        df = yf.download(ticker, start='2015-01-01', progress=False)
        aligned_close = df["Close"].iloc[-len(tr_labels):]
        aligned_open = df["Open"].iloc[-len(tr_labels):]
        aligned_volume = df["Volume"].iloc[-len(tr_labels):]  # ADD THIS
        
        # ... your existing prediction code ...
        
        pred_test = model.predict(tr_data, verbose=0)
        
        # Generate signals
        conf_th = 0.5
        res = directional_confidence_signals(pred_test, trend_window=3, conf_th=conf_th)
        
        # APPLY FILTERS HERE
        filter_results = apply_trend_filters(
            buy_mask=res['buy_mask'],
            sell_mask=res['sell_mask'],
            close_prices=aligned_close.to_numpy(),
            volumes=aligned_volume.to_numpy(),
            **get_filter_preset('moderate')
        )
        
        # Use filtered signals
        buy_mask = filter_results['buy_mask']
        sell_mask = filter_results['sell_mask']
        
        # Run backtest with filtered signals
        simulator = run_backtest_corrected(
            ticker=ticker,
            close_prices=aligned_close.to_numpy(),
            open_prices=aligned_open.to_numpy(),
            dates=aligned_close.index,
            buy_mask=buy_mask,
            sell_mask=sell_mask,
            initial_capital=initial_capital
        )
        
        # ... rest of your code stays the same ...
"""