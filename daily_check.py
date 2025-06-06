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

def load_model(path):
    # Search for the latest .keras or .h5 model file
    model_files = [os.path.join(path, f) for f in os.listdir(path) 
                   if f.endswith(".keras") or f.endswith(".h5")]
    
    if not model_files:
        raise FileNotFoundError("No .keras or .h5 model found in path.")

    latest_model = max(model_files, key=os.path.getmtime)
    model = tf.keras.models.load_model(
    latest_model,
    custom_objects={"Attention": Attention}
    )
    print(f"Loaded model from: {latest_model}")
    return model

if __name__ == "__main__":
    
    # 1.Fetch new day's stocks data
    # To run:
    fetcher = StockFetcher(base_path="/Users/admin/FinAi/market_data")
    fetcher.run()
    
    days_to_process = 230 # need at least 200 days to compute the moving avg + 30 to compute the last day's prediction
    doBalance = False

    # 2.Process data to extract features
    directory = '/Users/admin/FinAi/market_data/train'
    files = os.listdir(directory)
    # Get tickers from training
    train_symbols = training_model.get_symbols_from_folder(directory)
    
    tr_data, tr_labels, tr_symbols =  \
        extract_features_with_fft.extract_features_with_fft(train_symbols, directory, True, 'daily', days_to_process, doBalance)
        
    directory = '/Users/admin/FinAi/market_data/validation'
    files = os.listdir(directory)
    # Get tickers from training
    val_symbols = training_model.get_symbols_from_folder(directory)
    
    val_data, val_labels, val_symbols =  \
        extract_features_with_fft.extract_features_with_fft(val_symbols, directory, True, 'daily', days_to_process, doBalance)
        
    # Concatenate data and labels using NumPy
    all_data = np.concatenate((tr_data, val_data), axis=0)
    all_labels = np.concatenate((tr_labels, val_labels), axis=0)
    
    # Concatenate symbol lists using +
    all_symbols = np.concatenate((tr_symbols, val_symbols), axis=0)

    # 3. Load latest model
    model = load_model('/Users/admin/FinAi')
    
    # 4. Run model with latest data
    pred_test = model.predict(all_data)

    actions = []
    for probs in pred_test:
        margin_pred_classes = training_model.get_action_from_probs(probs, margin_threshold=0, prob_threshold=0)
        actions.append([margin_pred_classes])
        #print("Action:", ["0-HOLD", "1-SELL", "2-BUY"][margin_pred_classes])
        
    actions = np.array(actions)  # optional: convert to NumPy array
        
    # --- Settings ---
    threshold = 0.8
    margin_threshold = 0.2
    
    # --- Compute top-2 margins ---
    top2_sorted = np.sort(pred_test, axis=1)[:, -2:]
    margins = top2_sorted[:, 1] - top2_sorted[:, 0]
    
    # --- Get confident prediction indexes ---
    confident_idxs = np.where((np.max(pred_test, axis=1) > threshold) & (margins > margin_threshold))[0]
    
    # --- Extract predicted and true classes for confident samples ---
    confident_preds = np.argmax(pred_test[confident_idxs], axis=1)
    confident_symbols = all_symbols[confident_idxs]
    confident_perc = pred_test[confident_idxs]
    
    # Filter out neutral class (label=0)
    mask = (confident_preds != 0)

    confident_preds = confident_preds[mask]
    confident_symbols = confident_symbols[mask]
    confident_perc = confident_perc[mask]

    # Separate predictions and symbols by class
    preds_class_1 = confident_preds[confident_preds == 1]
    symbols_class_1 = confident_symbols[confident_preds == 1]
    perc_class_1 = confident_perc[confident_preds == 1]
    
    preds_class_2 = confident_preds[confident_preds == 2]
    symbols_class_2 = confident_symbols[confident_preds == 2]
    perc_class_2 = confident_perc[confident_preds == 2]
    
    # Unique symbols in each group
    unique_symbols_1 = np.unique(symbols_class_1)
    unique_symbols_2 = np.unique(symbols_class_2)
    
    # Plots   
    # Get predicted classes and their confidence values
    predicted_classes = np.argmax(pred_test, axis=1)
    confidence_scores = np.max(pred_test, axis=1)
    
    # Filter: Only keep Buy (1) and Sell (2) predictions
    mask = (predicted_classes == 1) | (predicted_classes == 2)
    filtered_classes = predicted_classes[mask]
    filtered_confidences = confidence_scores[mask]
    filtered_symbols = np.array(all_symbols)[mask]  # symbols must match pred_test in length
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(filtered_confidences)[::-1]
    sorted_classes = filtered_classes[sorted_indices]
    sorted_confidences = filtered_confidences[sorted_indices]
    sorted_symbols = filtered_symbols[sorted_indices]
    
    # Assign bar colors
    bar_colors = ['green' if cls == 2 else 'red' for cls in sorted_classes]
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(sorted_confidences))
    ax.bar(x, sorted_confidences, color=bar_colors)
    
    # Threshold line
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1, label=f'Threshold = {threshold:.2f}')
    
    # Labels and formatting
    ax.set_xlabel("Predictions (Sorted by Confidence)")
    ax.set_ylabel("Prediction Confidence")
    ax.set_title("Buy/Sell Predictions Sorted by Confidence")
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Optional: symbol labels
    # ax.set_xticks(x)
    # ax.set_xticklabels(sorted_symbols, rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Filter predictions above the threshold
    above_threshold_mask = sorted_confidences > threshold
    filtered_classes_th = sorted_classes[above_threshold_mask]
    filtered_confidences_th = sorted_confidences[above_threshold_mask]
    filtered_symbols_th = sorted_symbols[above_threshold_mask]
    
    # Assign bar colors for above threshold predictions
    bar_colors_th = ['green' if cls == 2 else 'red' for cls in filtered_classes_th]
    
    # Plot histogram with symbols above threshold
    fig, ax = plt.subplots(figsize=(16, 6))
    x_th = np.arange(len(filtered_confidences_th))
    ax.bar(x_th, filtered_confidences_th, color=bar_colors_th)
    
    # Add symbol labels on x-axis
    ax.set_xticks(x_th)
    ax.set_xticklabels(filtered_symbols_th, rotation=90, fontsize=8)
    
    ax.set_xlabel("Symbols (Confidence > Threshold)")
    ax.set_ylabel("Prediction Confidence")
    ax.set_title("Buy/Sell Predictions Above Confidence Threshold")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.show()



    
    
    
    
        