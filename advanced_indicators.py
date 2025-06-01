#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 15:42:44 2025

@author: Michele
"""

import numpy as np
import antropy as ant
from arch import arch_model

def add_garch_volatility(df):
    df = df.copy()
    df['GARCH_vol'] = np.nan

    if 'LogRet' not in df.columns:
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))

    returns = df['LogRet'].dropna() * 100  # Percent scale

    # Only fit if we have enough data (at least 30 non-NaNs is a safe bet)
    if len(returns) < 30:
        return df

    try:
        am = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        res = am.fit(disp='off')
        df.loc[returns.index, 'GARCH_vol'] = res.conditional_volatility / 100  # scale back
    except Exception as e:
        print(f"GARCH model failed: {e}")

    return df


def higuchi_fd(x, kmax=10):
    L = []
    N = len(x)
    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            Lmk = 0
            n_max = int(np.floor((N - m - 1) / k))
            for i in range(1, n_max):
                Lmk += abs(x[m + i*k] - x[m + (i-1)*k])
            Lmk = (Lmk * (N - 1) / (k * n_max * k))
            Lk.append(Lmk)
        L.append(np.mean(Lk))
    lnL = np.log(L)
    lnk = np.log(1./np.arange(1, kmax+1))
    # Linear fit
    coeffs = np.polyfit(lnk, lnL, 1)
    return coeffs[0]

def add_fractal_dimension(df):
    window = 30  # adjust window size
    fractal_dims = []
    for i in range(window, len(df)+1):
        x = df['Close'].iloc[i-window:i].values
        fd = higuchi_fd(x)
        fractal_dims.append(fd)
    # Ensure it matches the length of df
    padding = len(df) - len(fractal_dims)
    df['Fractal_Dimension'] = [np.nan] * padding + fractal_dims

    return df

def hurst_exponent(ts):
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

def add_hurst_exponent(df):
    window = 100
    hursts = []
    for i in range(window, len(df)+1):
        ts = df['Close'].iloc[i-window:i].values
        hurst = hurst_exponent(ts)
        hursts.append(hurst)
    padding = len(df) - len(hursts)
    df['Hurst_Exponent'] = [np.nan]*padding + hursts
    return df

def add_sample_entropy(df):
    window = 100
    samp_entropy = []
    for i in range(window, len(df)+1):
        ts = df['Close'].iloc[i-window:i].values
        se = ant.sample_entropy(ts)
        samp_entropy.append(se)
    padding = len(df) - len(samp_entropy)
    df['Sample_Entropy'] = [np.nan]*padding + samp_entropy
    return df

def add_advanced_features(df):
    df = add_garch_volatility(df)
    return df