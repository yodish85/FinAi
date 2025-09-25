#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:17:24 2025

@author: Michele
"""

import numpy as np
import os

# File paths
file_path = "/Users/admin/FinAi/train-val-data/test_filtered_data_20250811_170036.npy"
labels_path = "/Users/admin/FinAi/train-val-data/test_filtered_labels_20250811_170036.npy"
symbols_path = "/Users/admin/FinAi/train-val-data/test_filtered_symbols_20250811_170036.npy"

# Load arrays
data = np.load(file_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)
symbols = np.load(symbols_path, allow_pickle=True)

print(f"Data shape before filtering: {data.shape}")
print(f"Labels shape before filtering: {labels.shape}")
print(f"Symbols shape before filtering: {symbols.shape}")

# Filter out labels == 0
mask = labels != 0

filtered_data = data[mask]
filtered_labels = labels[mask]
filtered_symbols = symbols[mask]

print(f"Data shape after filtering: {filtered_data.shape}")
print(f"Labels shape after filtering: {filtered_labels.shape}")
print(f"Symbols shape after filtering: {filtered_symbols.shape}")

# Build new file paths with "filtered_" prefix
def make_filtered_path(path):
    folder, fname = os.path.split(path)
    return os.path.join(folder, "binary_" + fname)

filtered_data_path = make_filtered_path(file_path)
filtered_labels_path = make_filtered_path(labels_path)
filtered_symbols_path = make_filtered_path(symbols_path)

# Save filtered arrays
np.save(filtered_data_path, filtered_data)
np.save(filtered_labels_path, filtered_labels)
np.save(filtered_symbols_path, filtered_symbols)

print("Filtered files saved:")
print(filtered_data_path)
print(filtered_labels_path)
print(filtered_symbols_path)
