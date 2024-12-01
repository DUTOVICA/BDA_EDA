import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Daten laden
df = pd.read_csv('amazon_historical_prices_dynamic.csv')

# Konvertiere 'Date' in datetime-Format und sortiere nach Datum
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Nur das 'Close'-Feature für die Vorhersage verwenden
close_prices = df['Close'].values.reshape(-1, 1)

# Daten normalisieren
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Trainings- und Testdatensätze erstellen
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 60  # 60 Tage als Eingabe für die Vorhersage
X, y = create_sequences(scaled_data, sequence_length)

# Daten in Trainings- und Testsets aufteilen
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Daten für das LSTM-Modell anpassen
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# LSTM-Modell erstellen
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Modell kompilieren
model.compile(optimizer='adam', loss='mean_squared_error')

# Modell trainieren
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Modell evaluieren und Vorhersagen für die nächsten 5 Tage
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))

# Vorhersagen für die nächsten 5 Tage
predicted_prices = []
for _ in range(5):
    next_price = model.predict(last_sequence)
    predicted_prices.append(next_price[0, 0])  # Den Vorhersagewert speichern
    last_sequence = np.append(last_sequence[:, 1:, :], [[[next_price[0, 0]]]], axis=1)

# Vorhersagen zurück in den ursprünglichen Maßstab transformieren
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Ergebnisse visualisieren
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='Echte Preise')
future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=5)
plt.plot(future_dates, predicted_prices, label='Vorhergesagte Preise', linestyle='dashed', color='red')
plt.legend()
plt.show()
