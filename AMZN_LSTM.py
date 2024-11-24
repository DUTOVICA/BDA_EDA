import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from skopt import BayesSearchCV
import matplotlib.pyplot as plt

# Load the dataset (make sure it's in the same directory as the script or specify the path)
df = pd.read_csv('amazon_historical_prices_dynamic.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Preprocessing
df = df[['Close*']]  # We will only use the 'Close***' price as the feature

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Convert data to numerical values
scaled_data = pd.DataFrame(scaled_data, columns=["Close*"])

# Create sequences with a sliding window of 30 days
window_size = 1  # Using 30 days for input
target_size = 5  # We predict the next 5 days

# Create X (input) and y (target)
X = []
y = []

for i in range(window_size, len(scaled_data) - target_size):
    X.append(scaled_data['Close*'][i-window_size:i].values)  # Last 30 days as input
    y.append(scaled_data['Close*'][i:i+target_size].values)  # Next 5 days as output

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=target_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

# Initialize the model
model = LSTMModel(input_size=1, hidden_layer_size=64, output_size=target_size)

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, X_train, y_train, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_function(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Train the model
train_model(model, X_train, y_train, num_epochs=50)

# Test the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Evaluate the performance using Accuracy and Matthews Correlation Coefficient
y_pred = y_pred.numpy()
y_test = y_test.numpy()

# For MCC, we need binary classification, but we can treat it as a regression task and check for Close**ness
y_pred_binary = (y_pred > 0.5).astype(int)
y_test_binary = (y_test > 0.5).astype(int)

# Compute MCC
mcc = matthews_corrcoef(y_test_binary.flatten(), y_pred_binary.flatten())
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

# Function to optimize hyperparameters using Bayesian Optimization
def optimize_hyperparameters():
    param_space = {
        'hidden_layer_size': (32, 128),  # Min and Max for hidden layer size
        'lr': (1e-5, 1e-2),  # Learning rate range
        'batch_size': (32, 256)  # Batch size range
    }
    
    def objective(params):
        hidden_layer_size, lr, batch_size = params
        model = LSTMModel(input_size=1, hidden_layer_size=hidden_layer_size, output_size=target_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train the model
        train_model(model, X_train, y_train, num_epochs=10)
        
        # Test and calculate MSE as the objective function
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
        
        mse = mean_absolute_error(y_test, y_pred.numpy())
        return mse
    
    # Run Bayesian optimization
    optimizer = BayesSearchCV(objective, param_space, n_iter=10)
    optimizer.fit(X_train)
    return optimizer.best_params_

# Run hyperparameter optimization
best_params = optimize_hyperparameters()
print(f"Best Hyperparameters: {best_params}")

# Plot predictions vs true values
plt.figure(figsize=(10,6))
plt.plot(y_test.flatten(), label='True values')
plt.plot(y_pred.flatten(), label='Predicted values')
plt.legend()
plt.title('True vs Predicted Values')
plt.show()
