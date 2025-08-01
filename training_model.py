#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:21:11 2025

@author: Michele
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf

import os
import importlib
import datetime
import glob
import extract_features_with_fft
importlib.reload(extract_features_with_fft)
from sklearn.utils.class_weight import compute_class_weight
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf

print(tf.__version__)
print(tf.__file__)
tf.config.set_visible_devices([], 'GPU')
#tf.debugging.set_log_device_placement(True)

def get_subset_df(df, start_date, end_date, syms):
    
    subset_df = df[syms]
    subset_df = df[start_date:end_date]
    
    return subset_df

def norm_df(df):
    #df = df/df.max()
    #df = df/df.iloc[-1,:]
    df_mean = df.mean()
    df_std = df.std()
    return (df - df_mean) / df_std, df_mean, df_std

def plot_df(df, title):
    df.plot(title=title, fontsize = 14)
    plt.xlabel('Time')
    plt.ylabel(title)
    plt.show()
          
def plot_metric(metric, symbol, df):
    df[metric].plot()
    plt.title((metric + ' for ' + symbol))
    plt.show()
    plt.gca().invert_xaxis()
    plt.xlabel('Time')

def get_max_metric(metric, df):
    print("Max ", metric)
    return df[metric].max()

def to_python_type(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj

def compute_feature_minmax(dataframes_list, usecols, loadStats):
    """Compute global min and max per feature across all symbols."""
    
    if not loadStats:
        feature_mins = {}
        feature_maxs = {}

        for df in dataframes_list:
            df = df.dropna()
            for col in df.columns:
                if col == 'Date': continue
                values = df[col].dropna().to_numpy()
                min_val = np.min(values)
                max_val = np.max(values)
                feature_mins[col] = min(min_val, feature_mins.get(col, min_val))
                feature_maxs[col] = max(max_val, feature_maxs.get(col, max_val))

        with open('feature_mins.json', 'w') as f:
            json.dump({k: to_python_type(v) for k, v in feature_mins.items()}, f)
        
        with open('feature_maxs.json', 'w') as f:
            json.dump({k: to_python_type(v) for k, v in feature_maxs.items()}, f)

    else:
        with open('feature_mins.json') as f:
            feature_mins = json.load(f)
        with open('feature_maxs.json') as f:
            feature_maxs = json.load(f)

    return feature_mins, feature_maxs


def normalize_df_minmax(df, feature_mins, feature_maxs):
    df_norm = df.copy()
    for col in df.columns:
        if col in feature_mins and col != 'Date':
            min_val = feature_mins[col]
            max_val = feature_maxs[col]
            df_norm[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
    return df_norm


def compute_feature_stats(dataframes_list, usecols, loadStats):
    """Compute global mean and std per feature across all symbols."""
    
    if not loadStats:
        feature_sums = {}
        feature_squared_sums = {}
        feature_counts = {}
    
        for df in dataframes_list:
            #df = df.drop(columns=['Date'])  # drop Date so we only normalize numerical
            df = df.dropna()
            for col in df.columns:
                values = df[col].dropna().to_numpy()
                feature_sums[col] = feature_sums.get(col, 0) + np.sum(values)
                feature_squared_sums[col] = feature_squared_sums.get(col, 0) + np.sum(values ** 2)
                feature_counts[col] = feature_counts.get(col, 0) + len(values)
    
        feature_means = {col: feature_sums[col] / feature_counts[col] for col in feature_sums}
        feature_stds = {
            col: np.sqrt(feature_squared_sums[col] / feature_counts[col] - feature_means[col] ** 2)
            for col in feature_sums
        }
        with open('feature_means.json', 'w') as f:
            json.dump(feature_means, f)
        with open('feature_stds.json', 'w') as f:
            json.dump(feature_stds, f)
            
    else:
        with open('feature_means.json') as f:
            feature_means = json.load(f)
        with open('feature_stds.json') as f:
            feature_stds = json.load(f)
            
    return feature_means, feature_stds

def normalize_df_with_stats(df, feature_means, feature_stds):
    df_norm = df.copy()
    for col in df.columns:
        if col in feature_means and col != 'Date':
            df_norm[col] = (df[col] - feature_means[col]) / feature_stds[col]
    return df_norm

# Attention layer
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        self.attention_dense = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        score = self.attention_dense(inputs)  # (batch_size, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)  # Summing over time
        return context_vector

# Residual Conv block
def residual_conv_block(x, filters=16, kernel_size=5, dropout_rate=0.3):
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Ensure shapes match for residual connection
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding='same')(shortcut)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

# Model builder
def build_model(input_shape, num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Residual Conv block
    x = residual_conv_block(inputs, filters=8, kernel_size=4)

    # BiLSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3,
             kernel_regularizer=tf.keras.regularizers.l2(0.001))
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Attention
    #x = Attention()(x)

    # Dense layers
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def compute_fft(feature_data):
    fft_result = np.fft.fft(feature_data)
    real_part = np.real(fft_result)
    imag_part = np.imag(fft_result)

    # Normalize real and imaginary parts
    real_part_norm = (real_part - np.min(real_part)) / (np.max(real_part) - np.min(real_part) + 1e-8)
    imag_part_norm = (imag_part - np.min(imag_part)) / (np.max(imag_part) - np.min(imag_part) + 1e-8)

    return np.stack([real_part_norm, imag_part_norm], axis=1)  # shape: (n, 2)

@tf.keras.utils.register_keras_serializable(package='Custom', name='WeightedCategoricalCrossentropy')
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weights, reduction='sum_over_batch_size', name='weighted_categorical_crossentropy', **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        sample_weights = tf.reduce_sum(y_true * self.weights, axis=-1)
        return base_loss * sample_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'weights': self.weights.numpy().tolist()
        })
        return config

def train_model(train_data, train_labels, test_data, test_labels):
    
    print("TensorFlow version:", tf.__version__)
    print("Eager execution:", tf.executing_eagerly())
    
    # --- Inspect data types and structure ---
    print("Train data shape:", train_data.shape)
    print("Train data dtype:", train_data.dtype)
    print("Train data type:", type(train_data))
    print("Train data[0] type:", type(train_data[0]))
    print("Train data[0][0] type:", type(train_data[0][0]))
    print("Train data[0][0][0] type:", type(train_data[0][0][0]))
    
    print("Train labels shape:", train_labels.shape)
    print("Train labels dtype:", train_labels.dtype)
    print("Train labels type:", type(train_labels))
    print("Train labels[0] type:", type(train_labels[0]))
    
    # --- Filter NaNs ---
    nan_mask = ~np.isnan(train_data).any(axis=(1, 2))
    train_data = train_data[nan_mask]
    train_labels = train_labels[nan_mask]
    
    # --- Shuffle train/val data ---
    train_shuffle_idx = np.random.permutation(len(train_data))
    filtered_train_data = train_data[train_shuffle_idx]
    filtered_train_labels = train_labels[train_shuffle_idx]
    
    val_shuffle_idx = np.random.permutation(len(test_data))
    filtered_test_data = test_data[val_shuffle_idx]
    filtered_test_labels = test_labels[val_shuffle_idx]
    
    # --- Dataset construction ---
    batch_size = 16
    
    train_ds = tf.data.Dataset.from_tensor_slices((filtered_train_data, filtered_train_labels)) \
        .shuffle(buffer_size=len(train_data)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE) \
        .repeat()
    
    val_ds = tf.data.Dataset.from_tensor_slices((filtered_test_data, filtered_test_labels)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE) \
        .repeat()
    
    steps_per_epoch = len(train_data) // batch_size
    validation_steps = len(test_data) // batch_size
    
    # --- Model setup ---
    num_classes = 3
    input_shape = filtered_train_data.shape[1:]
    
    # Compute class weights
    if train_labels.ndim == 2:
        y_integers = np.argmax(train_labels, axis=1)
    else:
        y_integers = train_labels

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    print("Class weights:", dict(enumerate(class_weights_array)))
    
    # Use custom class-weighted categorical crossentropy
    loss_fn = WeightedCategoricalCrossentropy(class_weights_array)
    
    # --- Optimizer, loss, and model ---
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model = build_model(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # --- Callbacks ---
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # --- Train ---
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[early_stop]
    )

    # Evaluate on training
    scores = model.evaluate(filtered_train_data, filtered_train_labels, verbose=0)
    print(f"Train Accuracy: {scores[1]*100:.2f}% | Error: {100 - scores[1]*100:.2f}%")

    # Evaluate on test
    pred_test = model.predict(filtered_test_data)
    scores2 = model.evaluate(filtered_test_data, filtered_test_labels, verbose=0)
    print(f"Test Accuracy: {scores2[1]*100:.2f}% | Error: {100 - scores2[1]*100:.2f}%")

    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Confusion plot
    pred_classes = np.argmax(pred_test, axis=1)
    true_classes = np.argmax(filtered_test_labels, axis=1)
    plt.plot(pred_classes, label='Predicted')
    plt.plot(true_classes, label='True')
    plt.legend()
    plt.title("Predicted vs Actual Classes")
    plt.show()

    # Probability plot
    plt.plot(pred_test[:,0], label='Predicted Hold')
    plt.plot(pred_test[:,1], label='Predicted Buy')
    plt.plot(pred_test[:,2], label='Predicted Sell')
    plt.legend()
    plt.title("Probability of hold/buy/sell")
    plt.show()

    # Get current time and format it
    x = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Save the model with a timestamped filename
    model.save(f'/Users/admin/FinAi/{x}_model.keras')
    actions = []
    for probs in pred_test:
        margin_pred_classes = get_action_from_probs(probs, margin_threshold=0, prob_threshold=0)
        actions.append([margin_pred_classes])
        #print("Action:", ["HOLD", "BUY", "SELL"][margin_pred_classes])
        
    actions = np.array(actions)  # optional: convert to NumPy array
    
    true_classes = np.argmax(filtered_test_labels, axis=1)
    diff =  actions[:,0] - true_classes[:]
    plt.scatter(np.arange(len(diff)), diff, label='Diff', s=10)
    plt.legend()
    plt.title("Predicted vs Actual Classes")
    plt.show()
    
    # --- Settings ---
    threshold = 0.6
    margin_threshold = 0.2
    
    # --- Compute top-2 margins ---
    top2_sorted = np.sort(pred_test, axis=1)[:, -2:]
    margins = top2_sorted[:, 1] - top2_sorted[:, 0]
    
    # --- Get confident prediction indexes ---
    confident_idxs = np.where((np.max(pred_test, axis=1) > threshold) & (margins > margin_threshold))[0]
    
    # --- Extract predicted and true classes for confident samples ---
    confident_preds = np.argmax(pred_test[confident_idxs], axis=1)
    confident_trues = np.argmax(filtered_test_labels[confident_idxs], axis=1)
    
    # Filter out neutral class (label=0)
    mask = (confident_trues != 0) & (confident_preds != 0)

    confident_preds = confident_preds[mask]
    confident_trues = confident_trues[mask]

    diff_pred = confident_preds - confident_trues

    print("Confusion Matrix:")
    cm = confusion_matrix(confident_trues, confident_preds)
    print(cm)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sell", "Buy"], yticklabels=["Sell", "Buy"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # --- Plot prediction vs actual ---
    plt.figure(figsize=(12, 4))
    plt.plot(confident_preds, label='Predicted', marker='o', linestyle='--')
    plt.plot(confident_trues, label='True', marker='x', linestyle=':')
    plt.legend()
    plt.title('Confident Predictions vs Actual Labels')
    plt.xlabel('Instance Index')
    plt.ylabel('Class')
    plt.grid(True)
    plt.show()
    
    plt.scatter(np.arange(len(diff_pred)), diff_pred, label='Diff', s=10)
    plt.legend()
    plt.title("Predicted vs Actual Classes")
    plt.show()


def get_action_from_probs(probs, margin_threshold=0.3, prob_threshold=0.9):
    """
    Takes in an array of class probabilities and returns:
    - 0 for 'hold'
    - 1 for 'sell'
    - 2 for 'buy'
    
    Uses a margin threshold to filter out uncertain predictions.
    """
    sorted_probs = np.sort(probs)[::-1]  # sort descending
    top1 = sorted_probs[0]
    top2 = sorted_probs[1]
    margin = np.abs(top1 - top2)
    
    if margin > margin_threshold:
        idx = np.argmax(probs)
        if probs[idx] > prob_threshold:
            return idx  # Confident: return 'buy' or 'sell'
        else:
            return 0
    else:
        return 0  # Not confident enough: treat as 'hold'


def load_and_run_model(path, train_data, train_labels, test_data, test_labels):
    import daily_check

    from daily_check import load_model
    importlib.reload(daily_check)  # Reload the module, not the function
    model = daily_check.load_model(path)

    # Evaluate on training
    scores = model.evaluate(train_data, train_labels, verbose=0)
    print(f"Train Accuracy: {scores[1]*100:.2f}% | Error: {100 - scores[1]*100:.2f}%")

    # Evaluate on test
    pred_test = model.predict(test_data)
    scores2 = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test Accuracy: {scores2[1]*100:.2f}% | Error: {100 - scores2[1]*100:.2f}%")

    # Confusion plot
    pred_classes = np.argmax(pred_test, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    indices_1_or_2 = np.where((true_classes == 1) | (true_classes == 2))[0]

    plt.plot(pred_classes, label='Predicted')
    plt.plot(true_classes, label='True')
    plt.legend()
    plt.title("Predicted vs Actual Classes")
    plt.show()

    plt.plot(pred_classes[indices_1_or_2], label='Predicted')
    plt.plot(true_classes[indices_1_or_2], label='True')
    plt.legend()
    plt.title("Predicted vs Actual Classes (only buy and sell)")
    plt.show()
    
    actions = []
    for probs in pred_test:
        margin_pred_classes = get_action_from_probs(probs, margin_threshold=0, prob_threshold=0)
        actions.append([margin_pred_classes])
        #print("Action:", ["HOLD", "BUY", "SELL"][margin_pred_classes])
        
    actions = np.array(actions)  # optional: convert to NumPy array
    
    # --- Settings ---
    threshold = 0.75
    margin_threshold = 0.2
    
    # --- Compute top-2 margins ---
    top2_sorted = np.sort(pred_test, axis=1)[:, -2:]
    margins = top2_sorted[:, 1] - top2_sorted[:, 0]
    
    # --- Get confident prediction indexes ---
    confident_idxs = np.where((np.max(pred_test, axis=1) > threshold) & (margins > margin_threshold))[0]
    
    # --- Extract predicted and true classes for confident samples ---
    confident_preds = np.argmax(pred_test[confident_idxs], axis=1)
    confident_trues = np.argmax(test_labels[confident_idxs], axis=1)
    
    # Filter out neutral class (label=0)
    mask = (confident_trues != 0) & (confident_preds != 0)

    confident_preds = confident_preds[mask]
    confident_trues = confident_trues[mask]

    diff_pred = confident_preds - confident_trues

    print("Confusion Matrix:")
    cm = confusion_matrix(confident_trues, confident_preds)
    print(cm)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sell", "Buy"], yticklabels=["Sell", "Buy"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # --- Plot prediction vs actual ---
    plt.figure(figsize=(12, 4))
    plt.plot(confident_preds, label='Predicted', marker='o', linestyle='--')
    plt.plot(confident_trues, label='True', marker='x', linestyle=':')
    plt.legend()
    plt.title('Confident Predictions vs Actual Labels')
    plt.xlabel('Instance Index')
    plt.ylabel('Class')
    plt.grid(True)
    plt.show()
    
    plt.scatter(np.arange(len(diff_pred)), diff_pred, label='Diff', s=10)
    plt.legend()
    plt.title("Predicted vs Actual Classes")
    plt.show()

def load_datasets(directory):
    """
    Load the latest train/test data and labels from the directory.
    """
    train_data_path = find_latest_file(directory, "train", "data", "balanced")
    train_labels_path = find_latest_file(directory, "train", "labels", "balanced")
    test_data_path = find_latest_file(directory, "test", "data", "balanced")
    test_labels_path = find_latest_file(directory, "test", "labels", "balanced")
    
    # Assuming files are in .npy format; change as needed
    train_data = np.load(train_data_path, mmap_mode='r')
    train_labels = np.load(train_labels_path, mmap_mode='r')
    test_data = np.load(test_data_path, mmap_mode='r')
    test_labels = np.load(test_labels_path, mmap_mode='r')

    print(f"Loaded:\n  Train Data: {train_data_path}\n  Train Labels: {train_labels_path}")
    print(f"  Test Data: {test_data_path}\n  Test Labels: {test_labels_path}")
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=3)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=3)

    return train_data, train_labels, test_data, test_labels

def find_latest_file(directory, keyword1, keyword2, keyword3=None):
    """
    Find the most recent file in a directory containing all specified keywords.
    """
    files = [
        f for f in os.listdir(directory)
        if keyword1 in f and keyword2 in f and (keyword3 in f if keyword3 else True)
    ]
    if not files:
        keywords = f"'{keyword1}', '{keyword2}'" + (f", '{keyword3}'" if keyword3 else "")
        raise FileNotFoundError(f"No file found with keywords {keywords} in {directory}")
    
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)


def get_symbols_from_folder(base_dir):
    """
    Recursively find all .csv files under base_dir (any depth)
    and return their filenames (without extension) as a list of tickers.
    """
    # ** is “any number of directories”, recursive=True required
    pattern = os.path.join(base_dir, "**", "*.csv")
    csv_paths = glob.glob(pattern, recursive=True)
    symbols = [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]
    return symbols
        
if __name__ == "__main__":
    
    loadData = False
    loadModel = False
    days_to_process = []

    if not loadData:
        # training data
        directory = '/Users/admin/FinAi/market_data/train'
        files = os.listdir(directory)
        # Get tickers from training
        train_symbols = get_symbols_from_folder(directory)
        
        train_data, train_labels, train_symbols =  \
            extract_features_with_fft.extract_features_with_fft(train_symbols, directory, True, 'train', days_to_process)
        
        # validation data
        directory = '/Users/admin/FinAi/market_data/validation'
        # Get tickers from validation directory
        val_symbols   = get_symbols_from_folder(directory)

        extract_features_with_fft.extract_features_with_fft(val_symbols, directory, True, 'test', days_to_process)
        
        # And load data
        directory = "/Users/admin/FinAi/train-val-data"
        train_data, train_labels, test_data, test_labels = load_datasets(directory)
    else:
        # or just load data
        directory = "/Users/admin/FinAi/train-val-data"
        train_data, train_labels, test_data, test_labels = load_datasets(directory)
        
    
    if not loadModel:
        train_model(train_data, train_labels, test_data, test_labels)
    else:
        load_and_run_model('/Users/admin/FinAi', \
                  train_data, train_labels, test_data, test_labels)
