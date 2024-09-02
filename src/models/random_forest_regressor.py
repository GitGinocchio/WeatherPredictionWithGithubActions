from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np
import sys
import os

from utils.preprocessing import fetch_unique_data

data = fetch_unique_data('data/collected')
df : pd.DataFrame = pd.DataFrame(data)

def try_convert_numeric(val):
    try:
        return pd.to_numeric(val)
    except (ValueError, TypeError):
        return val

df = df.map(try_convert_numeric)

#print(df.dtypes)

area_encoder = LabelEncoder()
region_encoder =LabelEncoder()
country_encoder = LabelEncoder()
winddir_encoder = LabelEncoder()
weather_desc_encoder = LabelEncoder()

df['area_encoded'] = area_encoder.fit_transform(df['area'])
df['region_encoded'] = region_encoder.fit_transform(df['region'])
df['country_encoded'] = country_encoder.fit_transform(df['country'])
df['winddir16Point_encoded'] = winddir_encoder.fit_transform(df['winddir16Point'])
df['weatherDesc_encoded'] = weather_desc_encoder.fit_transform(df['weatherDesc'])

# Inputs: giorno, ora, mese, anno, area, country, region
# Outputs: tutto il resto

X = df[['minute','hour','day','month','year','latitude','longitude']] #'area_encoded','region_encoded','country_encoded','population' 'year'
y = df[["FeelsLikeC","cloudcover","humidity","precipMM","pressure","temp_C","uvIndex","visibility","windspeedKmph"]]

X_train, X_test, y_train, y_test = train_test_split(X,y)

#base_model = RandomForestRegressor()

#model = LinearRegression()

model = RandomForestRegressor(
    n_estimators=500,
    n_jobs=-1,
    verbose=0,
    criterion='absolute_error',
    oob_score=True
    )
#model = MultiOutputRegressor(base_model,n_jobs=-1)

#poly = PolynomialFeatures(degree=2)
#model = make_pipeline(poly, LinearRegression())

model.fit(X_train,y_train)

# Fai predizioni sui dati di test
y_pred = model.predict(X_test)

# Calcola le metriche di valutazione
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

new_data = pd.DataFrame({
    #'area_encoded': [3],  # Sostituisci con i tuoi dati reali
    #'region_encoded': [9],
    #'country_encoded': [7],
    'minute' : [0],
    'hour': [20],
    'day': [2],
    'month': [9],
    'year': [2024],
    #'population': [3000],
    'latitude': [45.467],
    'longitude': [9.200] #11.333
})

new_prediction = model.predict(new_data)
print("\nPredictions for new data:")
prediction_df = pd.DataFrame(new_prediction, columns=y.columns)
print(prediction_df)