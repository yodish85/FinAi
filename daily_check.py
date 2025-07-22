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
    
    from training_model import WeightedCategoricalCrossentropy

    model = tf.keras.models.load_model(
        latest_model,
        custom_objects={'WeightedCategoricalCrossentropy': WeightedCategoricalCrossentropy}
    )

    print(f"Loaded model from: {latest_model}")
    return model

if __name__ == "__main__":
    
    # 1.Fetch new day's stocks data
    # To run:
    fetcher = StockFetcher(base_path="/Users/admin/FinAi/market_data")
    fetcher.run()
    
    days_to_process = 240 # need at least 200 days to compute the moving avg + 40 to compute the last day's prediction
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
    #model = load_model('/Users/admin/FinAi/new model - 010725')
    
    # 4. Run model with latest data
    pred_test = model.predict(all_data)

    actions = []
    for probs in pred_test:
        margin_pred_classes = training_model.get_action_from_probs(probs, margin_threshold=0, prob_threshold=0)
        actions.append([margin_pred_classes])
        #print("Action:", ["0-HOLD", "1-SELL", "2-BUY"][margin_pred_classes])
        
    actions = np.array(actions)  # optional: convert to NumPy array
    
    # --- Settings ---
    threshold_buy = 0.74
    threshold_sell = 0.7
    margin_threshold = 0.2
    
    # --- Compute top-2 margins ---
    top2_sorted = np.sort(pred_test, axis=1)[:, -2:]
    margins = top2_sorted[:, 1] - top2_sorted[:, 0]
    
    # --- Predicted classes and confidence scores ---
    predicted_classes = np.argmax(pred_test, axis=1)
    confidence_scores = np.max(pred_test, axis=1)
    
    # --- Get confident prediction indexes with class-specific thresholds ---
    sell_mask = (predicted_classes == 1) & \
                (confidence_scores > threshold_sell) & \
                (margins > margin_threshold)
    
    buy_mask = (predicted_classes == 2) & \
               (confidence_scores > threshold_buy) & \
               (margins > margin_threshold)
    
    confident_idxs = np.where(sell_mask | buy_mask)[0]
    
    # --- Extract predicted and true classes for confident samples ---
    confident_preds = predicted_classes[confident_idxs]
    confident_symbols = np.array(all_symbols)[confident_idxs]
    confident_perc = pred_test[confident_idxs]
    
    # Filter out neutral class (label=0) if still present
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
    
    # --- Plot: All Buy/Sell predictions sorted by confidence ---
    mask = (predicted_classes == 1) | (predicted_classes == 2)
    filtered_classes = predicted_classes[mask]
    filtered_confidences = confidence_scores[mask]
    filtered_symbols = np.array(all_symbols)[mask]
    
    sorted_indices = np.argsort(filtered_confidences)[::-1]
    sorted_classes = filtered_classes[sorted_indices]
    sorted_confidences = filtered_confidences[sorted_indices]
    sorted_symbols = filtered_symbols[sorted_indices]
    
    bar_colors = ['green' if cls == 2 else 'red' for cls in sorted_classes]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(sorted_confidences))
    ax.bar(x, sorted_confidences, color=bar_colors)
    
    # Plot both thresholds
    ax.axhline(y=threshold_buy, color='green', linestyle='--', linewidth=1, label=f'Buy Threshold = {threshold_buy:.2f}')
    ax.axhline(y=threshold_sell, color='red', linestyle='--', linewidth=1, label=f'Sell Threshold = {threshold_sell:.2f}')
    
    # Show ticker labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_symbols, rotation=90, fontsize=8)
    
    ax.set_xlabel("Predictions (Sorted by Confidence)")
    ax.set_ylabel("Prediction Confidence")
    ax.set_title("Buy/Sell Predictions Sorted by Confidence")
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # --- Filter predictions above thresholds ---
    above_threshold_mask = ((sorted_classes == 1) & (sorted_confidences > threshold_sell)) | \
                           ((sorted_classes == 2) & (sorted_confidences > threshold_buy))
    
    filtered_classes_th = sorted_classes[above_threshold_mask]
    filtered_confidences_th = sorted_confidences[above_threshold_mask]
    filtered_symbols_th = sorted_symbols[above_threshold_mask]
    
    bar_colors_th = ['green' if cls == 2 else 'red' for cls in filtered_classes_th]
    
    # Separate symbols based on prediction class
    buy_symbols = [sym for sym, cls in zip(filtered_symbols_th, filtered_classes_th) if cls == 2]
    sell_symbols = [sym for sym, cls in zip(filtered_symbols_th, filtered_classes_th) if cls == 1]
    
    # --- Print Results ---
    print("Buy symbols:", ", ".join(buy_symbols))
    print("Sell/Short symbols:", ", ".join(sell_symbols))
    
    # --- Plot: Only confident Buy/Sell predictions ---
    fig, ax = plt.subplots(figsize=(16, 6))
    x_th = np.arange(len(filtered_confidences_th))
    bars = ax.bar(x_th, filtered_confidences_th, color=bar_colors_th)
    
    # Add ticker labels on x-axis
    ax.set_xticks(x_th)
    ax.set_xticklabels(filtered_symbols_th, rotation=90, fontsize=8)
    
    # Add prediction confidence as text on top of bars
    for i, conf in enumerate(filtered_confidences_th):
        ax.text(x_th[i], conf + 0.01, f"{conf:.2f}", ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Symbols (Above Class-Specific Thresholds)")
    ax.set_ylabel("Prediction Confidence")
    ax.set_title("Confident Buy/Sell Predictions")
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()