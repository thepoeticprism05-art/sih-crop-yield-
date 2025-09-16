import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------- Data Preparation ----------------------
rainfall_df = pd.read_csv('rainfall.csv')
yield_df = pd.read_csv('yield.csv')
pesticide_df = pd.read_csv('pesticide.csv')
ground_truth_df = pd.read_csv('ground_truth.csv')

# Merge datasets
merged_df = rainfall_df.merge(yield_df, on=['country', 'year'])
merged_df = merged_df.merge(pesticide_df, on=['country', 'year'])
merged_df = merged_df.merge(ground_truth_df, on=['country', 'year'])

# Rename columns
merged_df = merged_df.rename(columns={
    'yield_x': 'yield_rainfall',
    'yield_y': 'yield_pesticide',
    'yield': 'ground_truth_yield'
})

# Handle missing values
merged_df = merged_df.fillna(merged_df.mean())

# Feature selection
features = merged_df[['rainfall', 'pesticide', 'yield_rainfall', 'yield_pesticide']]
target = merged_df['ground_truth_yield']

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(features_scaled, target, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ---------------------- Random Forest Model ----------------------
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")

# Feature importance visualization
feature_importances = rf_model.feature_importances_
plt.barh(features.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# ---------------------- PyTorch Neural Network ----------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# Model, loss, and optimizer
input_dim = X_train.shape[1]
model = SimpleNN(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Final evaluation on test set
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
model.eval()
with torch.no_grad():
    y_pred_nn = model(X_test_tensor)
    rmse_nn = torch.sqrt(criterion(y_pred_nn, y_test_tensor)).item()
    print(f"Neural Network RMSE: {rmse_nn:.2f}")
