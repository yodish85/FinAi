# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
import os
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
    
def get_df_list(symbol_list, usecols=None):
    dataframes_list = []

    for symbol in symbol_list:
        path = f"/Users/admin/Desktop/financial_ai_model/data/{symbol}.csv"
        print(f"Reading: {path}")

        try:
            # Read just the header to get all available columns
            all_columns = pd.read_csv(path, nrows=0).columns.tolist()

            # Ensure 'Date' is included in usecols
            if usecols:
                final_usecols = ['Date'] + [col for col in usecols if col != 'Date']
            else:
                final_usecols = all_columns

            # Read CSV
            df = pd.read_csv(path,
                             usecols=final_usecols,
                             parse_dates=['Date'],
                             na_values=['nan'])

            df.set_index('Date', inplace=True)
            dataframes_list.append(df)

        except Exception as e:
            print(f"❌ Error reading {symbol}: {e}")

    return dataframes_list
        

import numpy as np
import pandas as pd
import scipy.signal
import json


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


def extract_features(symbol_list, plot, saveData, name, loadStats):
    usecols = ["Date", "Volume", "Adj Close", "High", "Low", "Open"]
    dataframes_list = get_df_list(symbol_list, usecols)
    
    # 1. Compute stats from training set
    feature_means, feature_stds = compute_feature_stats(dataframes_list, usecols, loadStats)
    
    # 2. Normalize each df
    df_norm_list = []
    for df in dataframes_list:
        df_norm = normalize_df_with_stats(df, feature_means, feature_stds)
        df_norm_list.append(df_norm)
    
    days = 30
    expected_shape = (days, len(usecols) - 1)
    train_data_list = []
    train_labels_list = []

    for counter, df in enumerate(df_norm_list):
        print(counter, '/', len(df_norm_list)) 
            
        for column in df:
            if 'Close' in column:
                df_tmp = df[column]

                # Find peaks in adj close
                sell_peaks, _ = scipy.signal.find_peaks(df_tmp, height=0, prominence=0.05)
                buy_peaks, _ = scipy.signal.find_peaks(-1 * df_tmp, prominence=0.05)

                labels = np.zeros(df_tmp.size)
                labels[sell_peaks] = 1
                labels[buy_peaks] = 2
                
                if plot:
                    # plot
                    df_tmp.plot(title='Avg Close', fontsize = 14)
                    plt.xlabel('Time')
                    plt.ylabel('Avg Close')
                    plt.plot(df_tmp[sell_peaks], 'rx')
                    plt.plot(df_tmp[buy_peaks], 'bo')
                    
                    plt.show()
                
                for ti in range(days, df.shape[0]):
                    window = []
                    # Add window with shape (days, features), reshape to (days, features, 1)
                    window = df.iloc[ti-days:ti].to_numpy()
                    #window_fft = fft_res_norm[ti-days:ti]
                    
                    # Add fft
                    #if window.shape[1] != window_fft.shape[1]:
                    #    print(f"Shape mismatch: norm={window.shape}, fft={window_fft.shape}")
                    #    continue  # or fix shape here
                    #window = np.concatenate((window, window_fft), axis=1)
                    
                    train_data_list.append(window)
                    if window.shape != expected_shape:
                        print(f"norm={window.shape}")
                        continue
                    # Labels
                    train_labels_list.append(labels[ti])

    # Stack all windows along the third dimension (axis=2)
    train_data = np.stack(train_data_list, axis=0) 
    train_labels = np.array(train_labels_list)  # shape: (samples,)
    
    # One-hot encode labels (3 classes: 0, 1, 2)
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=3)

    train_data = np.array(train_data, dtype=np.float32, copy=True)
    train_labels = np.array(train_labels, dtype=np.int32, copy=True)
    
    # solve class imbalance in train data
    cum_sum = np.sum(train_labels, axis=0)
    idxs_hold = np.where(train_labels[:,0] == 1)[0]
    idxs_buy = np.where(train_labels[:,1] == 1)[0]
    idxs_sell = np.where(train_labels[:,2] == 1)[0]
    idxs_hold = np.random.choice(idxs_hold, idxs_buy.shape)
    train_data = train_data[np.concatenate([idxs_hold, idxs_sell, idxs_buy])]
    train_labels = train_labels[np.concatenate([idxs_hold, idxs_sell, idxs_buy])]
    
    # remove nans
    # Find rows that do NOT contain any NaNs
    valid_rows = ~np.isnan(train_data).any(axis=(1,2))
    
    # Filter both X and y
    train_data_clean = train_data[valid_rows]
    train_labels_clean = train_labels[valid_rows]

    if saveData:
        # Save to NP
        np.save(name + "_data_all_fft_norm.npy", train_data_clean)
        np.save(name + "_labels_all_fft_norm.npy", train_labels_clean)
    
    return train_data_clean, train_labels_clean


def train_model(train_data, train_labels, test_data, test_labels):
    
    print("TensorFlow version:", tf.__version__)
    print("Eager execution:", tf.executing_eagerly())
    
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
    
    # Unique labels (needed for classifier output units)
    print("Unique labels:", np.unique(train_labels))

    #### train model ####
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(train_data.shape[1], train_data.shape[2])),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
    ])
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(train_data.shape[1], train_data.shape[2], 1)),

        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    """

    model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_data, train_labels,
        validation_split=0.5,
        epochs=100,
        callbacks=[early_stop]
    )
    
    # Evaluate on training
    scores = model.evaluate(train_data, train_labels, verbose=0)
    print(f"Train Accuracy: {scores[1]*100:.2f}% | Error: {100 - scores[1]*100:.2f}%")

    # Evaluate on test
    pred_test = model.predict(test_data)
    scores2 = model.evaluate(test_data, test_labels, verbose=0)
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
    true_classes = np.argmax(test_labels, axis=1)
    plt.plot(pred_classes, label='Predicted')
    plt.plot(true_classes, label='True')
    plt.legend()
    plt.title("Predicted vs Actual Classes")
    plt.show()

    
    # save model
    model.save('/Users/admin/Desktop/financial_ai_model/model.keras')


if __name__ == "__main__":
    
    loadData = True
    if not loadData:
        # training data
        directory = '/Users/admin/Desktop/financial_ai_model/training_data'
        files = os.listdir(directory)
        symbol_list = [];
        for fi in files:
            symbol_list.append(os.path.splitext(fi)[0])
        train_data, train_labels =  \
            extract_features(symbol_list, False, True, 'train', False)
        
        # validation data
        directory = '/Users/admin/Desktop/financial_ai_model/validation_data'
        files = os.listdir(directory)
        symbol_list = [];
        for fi in files:
            symbol_list.append(os.path.splitext(fi)[0])
        test_data, test_labels = extract_features(symbol_list, False, True, 'test', True)
    else:
        # or load data
        # To load later:
        train_data = np.load("train_data_all_fft_norm.npy")
        train_labels = np.load("train_labels_all_fft_norm.npy")
        test_data = np.load("test_data_all_fft_norm.npy")
        test_labels = np.load("test_labels_all_fft_norm.npy")
    
    train_model(train_data, train_labels, test_data, test_labels)
