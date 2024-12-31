from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch

from neural_networks.preprocessing import encode_label, apply_scaler
from neural_networks.models.lstm import LSTM
from neural_networks.training import train, test, validate
from utils.terminal import getlogger
from utils.db import Database

logger = getlogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"CUDA Available: {torch.cuda.is_available()}")
logger.info(f"Current Device: {device} (index: {device.index} type: {device.type})")

db = Database()

df = pd.read_sql("SELECT * FROM weather", db.connection)

logger.info(f"Data loaded successfully. Shape: {df.shape}")

logger.info(f"Encoding labels")
weather_desc_encoder = encode_label(df, "weatherDescription")
winddir16point_encoder = encode_label(df, "winddir16Point")
country_encoder = encode_label(df, "country")
region_encoder = encode_label(df, "region")
city_encoder = encode_label(df, "city")

logger.info(f"Applying scalers")
min_max_scaler = apply_scaler(df, ['year', 'month', 'day','hour', 'minute', 'latitude', 'longitude', "feelsLike", "cloudcover", "humidity", "precip", "pressure", "temp", "uvIndex", "visibility", "windspeed", "weatherDescription", "winddir16Point"], MinMaxScaler())
standard_scaler = apply_scaler(df, ['year', 'month', 'day','hour', 'minute', 'latitude', 'longitude', "feelsLike", "cloudcover", "humidity", "precip", "pressure", "temp", "uvIndex", "visibility", "windspeed", "weatherDescription", "winddir16Point"], StandardScaler())

X = df[['year', 'month', 'day','hour', 'minute', 'latitude', 'longitude']]
y = df[["feelsLike", "cloudcover", "humidity", "precip", "pressure", "temp", "uvIndex", "visibility", "windspeed", "weatherDescription", "winddir16Point"]]
#y = df[["feelsLike", "cloudcover", "humidity", "precip", "pressure", "temp", "uvIndex", "visibility", "windspeed"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

input_size = X_train_tensor.shape[1]
output_size = y_train_tensor.shape[1]
hidden_size = 128
num_layers = 3
learning_rate = 0.001
weight_decay = 1e-5
patience = 100
num_epochs = 10000
disable_patience = False

model = LSTM(input_size, output_size, hidden_size, num_layers).to(device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)
loss = nn.MSELoss()

train_loss, val_loss = train(
    train_dataloader=train_dataloader, 
    val_dataloader=test_dataloader, 
    model=model, 
    device=device, 
    loss_fn=loss, 
    optimizer=optimizer, 
    scheduler=scheduler, 
    num_epochs=num_epochs, 
    model_name="model.pth", 
    patience=patience,
    disable_patience=disable_patience,
    save_dir="./"
)


test_loss = test(
    model=model, 
    model_name="model.pth", 
    test_dataloader=test_dataloader, 
    device=device, 
    loss_fn=loss, 
    save_dir="./"
)
