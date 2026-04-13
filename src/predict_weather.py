from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from neural_networks.preprocessing import load_scaler
from neural_networks.prediction import predict
from neural_networks.models.lstm import LSTM
from datetime import datetime
import pandas as pd
import torch

input_size = 13
output_size = 9
hidden_size = 128
num_layers = 3
learning_rate = 0.001
weight_decay = 1e-5
patience = 100
num_epochs = 10000
disable_patience = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size, output_size, hidden_size, num_layers).to(device)

year = 2025
month = 3
day = 25
hour = 23
minute = 0
latitude = 45.467
longitude = 9.2

date_of_year = datetime(year, month, day).timetuple().tm_yday - 1 / 365 * 2 - 1
time_of_day = ((hour + minute) / 60) % 24
lat_to_eq = abs(latitude - 23.4368)
lon_to_me = abs(longitude)
lat_bend = int((latitude + 90) / 30)
lon_range = int((longitude / 60) % 2)

new_data = pd.DataFrame({
    'year': [year],
    'month': [month],
    'day': [day],
    'hour': [hour],
    'minute': [minute],
    'latitude': [latitude],
    'longitude': [longitude],
    'date_of_year' : [date_of_year], 
    'time_of_day' : [time_of_day], 
    'lat_to_eq' : [lat_to_eq], 
    'lon_to_me' : [lon_to_me],
    'lat_band' : [lat_bend], 
    'lon_range' : [lon_range]
})

predictions = predict(model, new_data.to_numpy())[0][0]

cloudcover_scaler = load_scaler('cloudcover')
feelsLike_scaler = load_scaler('feelsLike')
heat_index_scaler = load_scaler('heat_index')
humidity_scaler = load_scaler('humidity')
precip_scaler = load_scaler('precip')
pressure_scaler = load_scaler('pressure')
sky_index_scaler = load_scaler('sky_index')
temp_scaler = load_scaler('temp')
uvIndex_scaler = load_scaler('uvIndex')
visibility_scaler = load_scaler('visibility')
windspeed_scaler = load_scaler('windspeed')
weatherIndex_scaler = load_scaler('weather_index')

[
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
]

feelsLike = feelsLike_scaler.inverse_transform([[predictions[0]]])[0][0]
cloudcover = cloudcover_scaler.inverse_transform([[predictions[1]]])[0][0]
humidity = humidity_scaler.inverse_transform([[predictions[2]]])[0][0]
precip = precip_scaler.inverse_transform([[predictions[3]]])[0][0]
pressure = pressure_scaler.inverse_transform([[predictions[4]]])[0][0]
temp = temp_scaler.inverse_transform([[predictions[5]]])[0][0]
uvIndex = uvIndex_scaler.inverse_transform([[predictions[6]]])[0][0]
visibility = visibility_scaler.inverse_transform([[predictions[7]]])[0][0]
windspeed = windspeed_scaler.inverse_transform([[predictions[8]]])[0][0]
#weatherDescription = weatherDescription_scaler.inverse_transform([[predictions[9]]])[0][0]
#winddir16Point = winddir16Point_scaler.inverse_transform([[predictions[10]]])[0][0]
#heat_index = heat_index_scaler.inverse_transform([[predictions[9]]])[0][0]
#weather_index = weatherIndex_scaler.inverse_transform([[predictions[10]]])[0][0]
#sky_index = sky_index_scaler.inverse_transform([[predictions[11]]])[0][0]

print(f'{float(feelsLike)=}')
print(f'{float(cloudcover)=}')
print(f'{float(precip)=}')
print(f'{float(pressure)=}')
print(f'{float(temp)=}')
print(f'{float(uvIndex)=}')
print(f'{float(visibility)=}')
print(f'{float(windspeed)=}')
#print(f'{float(heat_index)=}')
#print(f'{float(weather_index)=}')
#print(f'{float(sky_index)=}')


