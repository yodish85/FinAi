import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler  # <-- Add this import

# ----------------------------
# Download Data
# ----------------------------
ticker = 'AAPL'
print(f"🔄 Downloading data for {ticker}...")
df = yf.download(ticker, start='2015-01-01', end='2024-12-31')

# ----------------------------
# Flatten MultiIndex columns
# ----------------------------
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
print(f"✅ Data downloaded: {df.shape}")

# ----------------------------
# Select columns
# ----------------------------
# If MultiIndex, flatten it
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]

# Handle single-level columns that may already have ticker suffixes
expected_cols = [f'{col}_{ticker}' for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
raw_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

if all(col in df.columns for col in expected_cols):
    df = df[expected_cols]
elif all(col in df.columns for col in raw_cols):
    df = df[raw_cols]
    df.columns = [f"{col}_{ticker}" for col in df.columns]
else:
    raise KeyError("❌ Could not find expected columns in DataFrame. Columns available: " + str(df.columns.tolist()))


# ----------------------------
# Feature Engineering
# ----------------------------
print("⚙️  Generating features...")
df['Return'] = df[f'Close_{ticker}'].pct_change()
df['LogReturn'] = np.log(df[f'Close_{ticker}'] / df[f'Close_{ticker}'].shift(1))
df['MA_5'] = df[f'Close_{ticker}'].rolling(5).mean()
df['MA_10'] = df[f'Close_{ticker}'].rolling(10).mean()
df['STD_5'] = df[f'Close_{ticker}'].rolling(5).std()
df['Volume_Change'] = df[f'Volume_{ticker}'].pct_change()
df.dropna(inplace=True)
print(f"✅ Feature engineering complete. Data shape: {df.shape}")

# ----------------------------
# Target Variable: Future Log Return
# ----------------------------
df['FutureLogReturn'] = df['LogReturn'].shift(-1)

if 'FutureLogReturn' not in df.columns:
    raise KeyError("❌ 'FutureLogReturn' column is missing!")

df.dropna(subset=['FutureLogReturn'], inplace=True)
print("✅ 'FutureLogReturn' created and cleaned.")

# ----------------------------
# Classification Labels
# ----------------------------
bins = [-np.inf, -0.005, 0.005, np.inf]
labels = [0, 1, 2]  # Down, Neutral, Up
df['Direction'] = pd.cut(df['FutureLogReturn'], bins=bins, labels=labels)
df.dropna(subset=['Direction'], inplace=True)
df['Direction'] = df['Direction'].astype(int)
print("✅ Classification labels assigned.")

# ----------------------------
# Sequence Creation
# ----------------------------
def create_windowed_dataset(df, window=60):
    features = [f'Open_{ticker}', f'High_{ticker}', f'Low_{ticker}', f'Close_{ticker}', f'Volume_{ticker}',
                'Return', 'LogReturn', 'MA_5', 'MA_10', 'STD_5', 'Volume_Change']
    X, y = [], []

    for i in range(window, len(df)):
        window_data = df[features].iloc[i - window:i]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(window_data)
        X.append(scaled)
        y.append(df['Direction'].iloc[i])

    return np.array(X), np.array(y)

X, y = create_windowed_dataset(df)
print("✅ Sequences created. Shape:", X.shape)

# ----------------------------
# Oversample to balance classes
# ----------------------------
# Reshape X to (n_samples, -1) to use RandomOverSampler
X_flat = X.reshape((X.shape[0], -1))
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y)

# Reshape back to (samples, time steps, features)
X_balanced = X_resampled.reshape((-1, X.shape[1], X.shape[2]))
y_balanced = y_resampled

# One-hot encode the balanced labels
y_cat = tf.keras.utils.to_categorical(y_balanced, num_classes=3)

# ----------------------------
# Train-Test Split (use balanced data)
# ----------------------------
split = int(0.8 * len(X_balanced))
X_train, X_val = X_balanced[:split], X_balanced[split:]
y_train, y_val = y_cat[:split], y_cat[split:]

# ----------------------------
# Build Model
# ----------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------
# Train Model
# ----------------------------
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=['Down', 'Neutral', 'Up']))

cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Neutral', 'Up'], yticklabels=['Down', 'Neutral', 'Up'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# Directional Accuracy
# ----------------------------
def to_direction(x):
    return -1 if x == 0 else (1 if x == 2 else 0)

direction_true = [to_direction(i) for i in y_true_classes]
direction_pred = [to_direction(i) for i in y_pred_classes]

print("📈 Directional Accuracy:", accuracy_score(direction_true, direction_pred))
