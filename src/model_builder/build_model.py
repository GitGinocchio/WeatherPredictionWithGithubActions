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

from utils import fetch_unique_data



data = fetch_unique_data('data/collected')
df = pd.DataFrame(data)

label_encoder = LabelEncoder()

df['area_encoded'] = label_encoder.fit_transform(df['area'])
df['region_encoded'] = label_encoder.fit_transform(df['region'])
df['country_encoded'] = label_encoder.fit_transform(df['country'])
df['winddir16Point_encoded'] = label_encoder.fit_transform(df['winddir16Point'])
df['weatherDesc_encoded'] = label_encoder.fit_transform(df['weatherDesc'])

#print(df[['area','area_encoded','region','region_encoded','country','country_encoded','latitude','longitude','population']].tail(50))

# Inputs: giorno, ora, mese, anno, area, country, region
# Outputs: tutto il resto

X = df[['minute','hour','day','month','year','latitude','longitude']] #'area_encoded','region_encoded','country_encoded','population'
y = df.drop(columns=['minute','precipInches','pressureInches','winddirDegree','visibilityMiles','windspeedMiles','weatherCode','FeelsLikeF','temp_F','area','region','country','winddir16Point','hour','month','day','year',"area_encoded","region_encoded","country_encoded","winddir16Point_encoded","weatherDesc","weatherDesc_encoded",'localObsDateTime','population','latitude','longitude'])



X_train, X_test, y_train, y_test = train_test_split(X,y)

#base_model = RandomForestRegressor()
#model = MultiOutputRegressor(base_model)

#model = LinearRegression()

model = RandomForestRegressor()

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
    'minute' : [30],
    'hour': [21],
    'day': [31],
    'month': [8],
    'year': [2024],
    #'population': [3000],
    'latitude': [44.417],
    'longitude': [8.950] #11.333
})

new_prediction = model.predict(new_data)
print("\nPredictions for new data:")
print(pd.DataFrame(new_prediction, columns=y.columns))
