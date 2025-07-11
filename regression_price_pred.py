import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

ticker = 'AAPL'
df = yf.download(ticker, start='2015-01-01', end='2024-12-31', group_by='ticker')

# Flatten MultiIndex if it exists
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    print("✅ Columns flattened:", df.columns.tolist())

# ----------------------------
# Feature Engineering
# ----------------------------

print("⚙️  Generating features...")

# Define column names based on ticker
close_col = f'{ticker}_Close'
high_col = f'{ticker}_High'
low_col = f'{ticker}_Low'
volume_col = f'{ticker}_Volume'

# Basic features
df['Return'] = df[close_col].pct_change()
df['LogReturn'] = np.log(df[close_col] / df[close_col].shift(1))
df['MA_5'] = df[close_col].rolling(window=5).mean()

# Advanced features (make sure these are created before using them)
df['MA_10'] = df[close_col].rolling(window=10).mean()
df['STD_5'] = df[close_col].rolling(window=5).std()
df['Volume_Change'] = df[volume_col].pct_change()
df['Price_Range'] = df[high_col] - df[low_col]
df['Momentum_3'] = df[close_col] - df[close_col].shift(3)

# Drop rows with NaN values from rolling calculations
df.dropna(inplace=True)

# Define target variable: Direction (1 if price goes up, 0 if down)
df['Direction'] = (df[close_col].shift(-1) > df[close_col]).astype(int)

print("✅ Features generated. Columns now:", df.columns.tolist())

# ----------------------------
# Target
# ----------------------------
df['FutureLogReturn'] = df['LogReturn'].shift(-1)
df.dropna(subset=['FutureLogReturn'], inplace=True)

threshold = df['LogReturn'].rolling(20).std().mean()
print(f"ℹ️ Classification threshold (log return): {threshold:.5f} ({(np.exp(threshold)-1)*100:.2f}% movement)")

bins = [-np.inf, -threshold, threshold, np.inf]
df['Direction'] = pd.cut(df['FutureLogReturn'], bins=bins, labels=[0, 1, 2]).astype(int)
print("✅ Targets assigned.")

from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df['Direction'] == 1]
df_minority_0 = df[df['Direction'] == 0]
df_minority_2 = df[df['Direction'] == 2]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    
                                   n_samples=min(len(df_minority_0), len(df_minority_2)), 
                                   random_state=42)

# Combine
df_balanced = pd.concat([df_majority_downsampled, df_minority_0, df_minority_2])


# ----------------------------
# Create Sequences
# ----------------------------
def create_windowed_dataset(df, window=100):
    features = [
    f'{ticker}_Open', f'{ticker}_High', f'{ticker}_Low', f'{ticker}_Close', f'{ticker}_Volume',
    'MA_10', 'STD_5', 'Volume_Change', 'Price_Range', 'Momentum_3'
]

    X, y = [], []
    for i in range(window, len(df)):
        window_data = df[features].iloc[i - window:i]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(window_data)
        X.append(scaled)
        y.append(df['Direction'].iloc[i])
    return np.array(X), np.array(y)

X, y = create_windowed_dataset(df_balanced)
X, y = shuffle(X, y, random_state=42)

print("✅ Windowed dataset. Shape:", X.shape)

# ----------------------------
# Train-Test Split
# ----------------------------
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# ----------------------------
# Oversample with RandomOverSampler
# ----------------------------
X_train_flat = X_train.reshape((X_train.shape[0], -1))
ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train_flat, y_train)
X_train_over = X_train_over.reshape((-1, X.shape[1], X.shape[2]))

print("✅ Oversampling complete. New shape:", X_train_over.shape)

# ----------------------------
# One-hot Encode Labels
# ----------------------------
y_train_cat = tf.keras.utils.to_categorical(y_train_over, num_classes=3)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=3)

# ----------------------------
# Build Model
# ----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------
# Train
# ----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_over, y_train_cat,
    epochs=500,
    batch_size=128,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stop],
    verbose=1
)


# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val_cat, axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    labels=[0, 1, 2],
    target_names=['Down', 'Neutral', 'Up'],
    zero_division=0
))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Neutral', 'Up'], yticklabels=['Down', 'Neutral', 'Up'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# Directional Accuracy
# ----------------------------
def to_direction(x): return -1 if x == 0 else (1 if x == 2 else 0)
print("📈 Directional Accuracy:", accuracy_score(
    [to_direction(i) for i in y_true_classes],
    [to_direction(i) for i in y_pred_classes]
))

# ----------------------------
# Fresh Fetch & Feature Generation (Fixed)
# ----------------------------
print("🔄 Fetching fresh data...")

# Fetch last 180 calendar days
df_live = yf.download(ticker, period="180d")

# Generate features using standard column names
df_live['Return'] = df_live['Close'].pct_change()
df_live['LogReturn'] = np.log(df_live['Close'] / df_live['Close'].shift(1))
df_live['MA_5'] = df_live['Close'].rolling(window=5).mean()
df_live['MA_10'] = df_live['Close'].rolling(window=10).mean()
df_live['STD_5'] = df_live['Close'].rolling(window=5).std()
df_live['Volume_Change'] = df_live['Volume'].pct_change()
df_live['Price_Range'] = df_live['High'] - df_live['Low']
df_live['Momentum_3'] = df_live['Close'] - df_live['Close'].shift(3)

# Drop rows with NaNs
df_live.dropna(inplace=True)

print("✅ Fresh features computed.")

# ----------------------------
# Extract Latest Window
# ----------------------------
window = 100
latest_window = df_live[-window:]

features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA_10', 'STD_5', 'Volume_Change', 'Price_Range', 'Momentum_3'
]

# Scale and reshape
scaler = MinMaxScaler()
latest_scaled = scaler.fit_transform(latest_window[features])
latest_scaled = latest_scaled.reshape(1, latest_scaled.shape[0], latest_scaled.shape[1])

# ----------------------------
# Predict
# ----------------------------
next_day_pred = model.predict(latest_scaled)
predicted_class = np.argmax(next_day_pred)

class_map = {0: "📉 Down", 1: "➖ Neutral", 2: "📈 Up"}
print(f"\n🧠 Predicted next day movement for {ticker}: {class_map[predicted_class]}")

