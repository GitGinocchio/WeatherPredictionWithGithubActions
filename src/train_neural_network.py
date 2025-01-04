from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch

from neural_networks.preprocessing import \
    encode_label,                         \
    apply_scaler,                         \
    create_feature,                       \
    apply_onehot_encoder
from neural_networks.models.lstm import LSTM
from neural_networks.training import train, test
from utils.terminal import getlogger
from utils.db import Database

logger = getlogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"CUDA Available: {torch.cuda.is_available()}")
logger.info(f"Current Device: {device} (index: {device.index} type: {device.type})")

db = Database()

weather_df = pd.read_sql("SELECT * FROM weather", db.connection)
hourly_df = pd.read_sql("""
                        SELECT year, 
                               month,
                               day,
                               time AS hour,
                               0 AS minute,
                               temp,
                               feelsLike,
                               tempF,
                               feelsLikeF,
                               cloudcover,
                               humidity,
                               uvIndex,
                               precip,
                               precipInches,
                               pressure,
                               pressureInches,
                               visibility,
                               visibilityMiles,
                               weatherCode,
                               weatherDescription,
                               winddir,
                               winddir16Point,
                               windspeed,
                               windspeedMiles,
                               "" as city,
                               "" as country,
                               "" as region,
                               latitude,
                               longitude,
                               0 AS population
                        FROM hourly""", 
                        db.connection)

df = pd.concat([weather_df, hourly_df])

logger.info(f"Data loaded successfully. Shape: {df.shape}")

logger.info(f"Encoding labels")
weather_desc_encoder = encode_label(df, "weatherDescription")
winddir16point_encoder = encode_label(df, "winddir16Point")
country_encoder = encode_label(df, "country")
region_encoder = encode_label(df, "region")
city_encoder = encode_label(df, "city")

logger.info(f"Creating new Features")
create_feature(df, "heat_index", lambda row: (row["feelsLike"] + row["temp"]) / 2 * row["humidity"])
create_feature(df,"weather_index", lambda row: (row["precip"] + row["pressure"] + row["windspeed"]) / 3)
create_feature(df, "sky_index", lambda row: (row["cloudcover"] * row["visibility"] + row["uvIndex"]) / 2)

create_feature(df, "date_of_year", lambda row: (datetime(int(row["year"]), int(row["month"]), int(row["day"])).timetuple().tm_yday - 1) / 365 * 2 - 1)
create_feature(df, "time_of_day", lambda row: ((row["hour"] + row["minute"]) / 60) % 24)
create_feature(df, "lat_to_eq", lambda row: abs(row["latitude"] - 23.4368))
create_feature(df, "lon_to_me", lambda row: abs(row["longitude"]))
create_feature(df, "lat_band", lambda row: int((row["latitude"] + 90) / 30))
create_feature(df, "lon_range", lambda row: int((row["longitude"] / 60) % 2))

logger.info(f"Applying scalers")

min_max_scaler = apply_scaler(df, 
    [
    #"date_of_year", "time_of_day", 
    #"lat_to_eq", "lon_to_me",
    #"lat_band", "lon_range",
    "heat_index", "weather_index", "sky_index", 
    "feelsLike", "cloudcover", "humidity", "precip", "pressure", "temp", 
    "uvIndex", "visibility", "windspeed", "weatherDescription", "winddir16Point"], 
    MinMaxScaler()
)
#standard_scaler = apply_scaler(df, ["feelsLike", "cloudcover", "humidity", "precip", "pressure", "temp", "uvIndex", "visibility", "windspeed", "weatherDescription", "winddir16Point"], StandardScaler())

X = df[[
    'year', 
    'month', 
    'day',
    'hour', 
    'minute', 
    'latitude', 
    'longitude', 
    'date_of_year', 
    'time_of_day', 
    'lat_to_eq', 
    #'lon_to_me',
    #'lat_band', 
    'lon_range'
]]

y = df[[
    "feelsLike", 
    "cloudcover", 
    "humidity", 
    "precip", 
    "pressure", 
    "temp", 
    "uvIndex", 
    "visibility", 
    "windspeed", 
    #"weatherDescription",
    #"winddir16Point", 
    "heat_index", 
    "weather_index", 
    "sky_index"
]]

input("Press Enter to continue...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

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
optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
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
