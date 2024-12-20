
NEW_WEATHER_CONDITION_QUERY = """
INSERT INTO weather (
    localObsDateTime,
    
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
    
    city,
    country,
    region,
    latitude,
    longitude,
    population
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

NEW_DAILY_QUERY = """
INSERT INTO daily (
    date,
    
    latitude,
    longitude,

    moon_illumination,
    moon_phase,
    
    moonrise,
    moonset,
    
    sunrise,
    sunset,

    avgTemp,
    avgTempF,

    maxTemp,
    maxTempF,

    minTemp,
    minTempF,

    sunHour,
    totalSnow,
    uvIndex
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

NEW_HOURLY_QUERY = """
INSERT INTO hourly (
    date,
    time,
    
    latitude,
    longitude,

    dewPoint,
    DewPointF,

    feelsLike,
    feelsLikeF,

    heatIndex,
    heatIndexF,

    windChill,
    windChillF,

    windGust,
    windGustMiles,

    chanceoffog,
    chanceoffrost,
    chanceofhightemp,
    chanceofovercast,
    chanceofrain,
    chanceofremdry,
    chanceofsnow,
    chanceofsunshine,
    chanceofthunder,
    chanceofwindy,

    shortRad,
    diffRad,

    cloudcover,
    humidity,
    uvIndex,

    precip,
    precipInches,

    pressure,
    pressureInches,

    temp,
    tempF,

    visibility,
    visibilityMiles,

    weatherCode,
    weatherDescription,

    windspeed,
    windspeedMiles,

    winddir,
    winddir16Point
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

HAS_WEATHER_CONDITION = """SELECT * FROM weather WHERE latitude = ? AND longitude = ? AND localObsDateTime = ?"""

GET_ALL_WEATHER_CONDITIONS = """SELECT * FROM weather"""

HAS_DAILY = """SELECT * FROM daily WHERE latitude = ? AND longitude = ? AND date = ?"""

REMOVE_LOCAL_OBS_TIME = """
-- Disabilita temporaneamente le chiavi esterne
PRAGMA foreign_keys = OFF;

-- 1. Ricrea la tabella `daily` senza `localObsDateTime` e con il nuovo vincolo
CREATE TABLE daily_new (
    date DATETIME,
    latitude FLOAT,
    longitude FLOAT,
    moon_illumination INT,
    moon_phase TEXT,
    moonrise TEXT,
    moonset TEXT,
    sunrise TEXT,
    sunset TEXT,
    avgTemp FLOAT,
    avgTempF FLOAT,
    maxTemp INT,
    maxTempF INT,
    minTemp INT,
    minTempF INT,
    sunHour FLOAT,
    totalSnow FLOAT,
    uvIndex INT,
    PRIMARY KEY (date, latitude, longitude),
    FOREIGN KEY (latitude, longitude) REFERENCES weather (latitude, longitude)
);

-- 2. Copia i dati dalla vecchia tabella alla nuova tabella
INSERT INTO daily_new (
    date, latitude, longitude, moon_illumination, moon_phase, moonrise, moonset, 
    sunrise, sunset, avgTemp, avgTempF, maxTemp, maxTempF, minTemp, minTempF, 
    sunHour, totalSnow, uvIndex
)
SELECT
    date, latitude, longitude, moon_illumination, moon_phase, moonrise, moonset, 
    sunrise, sunset, avgTemp, avgTempF, maxTemp, maxTempF, minTemp, minTempF, 
    sunHour, totalSnow, uvIndex
FROM daily;

-- 3. Elimina la vecchia tabella e rinomina la nuova
DROP TABLE daily;
ALTER TABLE daily_new RENAME TO daily;

-- 4. Ripeti il processo per `hourly`
CREATE TABLE hourly_new (
    date DATETIME,
    time INT,
    latitude FLOAT,
    longitude FLOAT,
    dewPoint INT,
    DewPointF INT,
    feelsLike INT,
    feelsLikeF INT,
    heatIndex INT,
    heatIndexF INT,
    windChill INT,
    windChillF INT,
    windGust INT,
    windGustMiles INT,
    chanceoffog INT,
    chanceoffrost INT,
    chanceofhightemp INT,
    chanceofovercast INT,
    chanceofrain INT,
    chanceofremdry INT,
    chanceofsnow INT,
    chanceofsunshine INT,
    chanceofthunder INT,
    chanceofwindy INT,
    shortRad FLOAT,
    diffRad FLOAT,
    cloudcover INT,
    humidity INT,
    uvIndex INT,
    precip FLOAT,
    precipInches FLOAT,
    pressure INT,
    pressureInches INT,
    temp INT,
    tempF INT,
    visibility INT,
    visibilityMiles INT,
    weatherCode INT,
    weatherDescription TEXT,
    windspeed INT,
    windspeedMiles INT,
    winddir INT,
    winddir16Point TEXT,
    PRIMARY KEY (date, time, latitude, longitude),
    FOREIGN KEY (latitude, longitude) REFERENCES weather (latitude, longitude)
);

-- Copia i dati
INSERT INTO hourly_new (
    date, time, latitude, longitude, dewPoint, DewPointF, feelsLike, feelsLikeF,
    heatIndex, heatIndexF, windChill, windChillF, windGust, windGustMiles, 
    chanceoffog, chanceoffrost, chanceofhightemp, chanceofovercast, chanceofrain,
    chanceofremdry, chanceofsnow, chanceofsunshine, chanceofthunder, chanceofwindy,
    shortRad, diffRad, cloudcover, humidity, uvIndex, precip, precipInches, pressure,
    pressureInches, temp, tempF, visibility, visibilityMiles, weatherCode, weatherDescription,
    windspeed, windspeedMiles, winddir, winddir16Point
)
SELECT
    date, time, latitude, longitude, dewPoint, DewPointF, feelsLike, feelsLikeF,
    heatIndex, heatIndexF, windChill, windChillF, windGust, windGustMiles, 
    chanceoffog, chanceoffrost, chanceofhightemp, chanceofovercast, chanceofrain,
    chanceofremdry, chanceofsnow, chanceofsunshine, chanceofthunder, chanceofwindy,
    shortRad, diffRad, cloudcover, humidity, uvIndex, precip, precipInches, pressure,
    pressureInches, temp, tempF, visibility, visibilityMiles, weatherCode, weatherDescription,
    windspeed, windspeedMiles, winddir, winddir16Point
FROM hourly;

-- Elimina la vecchia tabella e rinomina
DROP TABLE hourly;
ALTER TABLE hourly_new RENAME TO hourly;

-- Riattiva le chiavi esterne
PRAGMA foreign_keys = ON;
"""

HUMIDITY_FIX = """
-- Disabilita temporaneamente le chiavi esterne
PRAGMA foreign_keys = OFF;

-- 1. Ricrea la tabella `weather` con la colonna `humidity` corretta
CREATE TABLE weather_new (
    localObsDateTime DATETIME,

    temp INT,
    feelsLike INT,

    tempF INT,
    feelsLikeF INT,

    cloudcover INT,
    humidity INT,
    uvIndex INT,

    precip FLOAT,
    precipInches FLOAT,

    pressure INT,
    pressureInches INT,

    visibility INT,
    visibilityMiles INT,

    weatherCode INT,
    weatherDescription TEXT,

    winddir INT,
    winddir16Point TEXT,
    
    windspeed INT,
    windspeedMiles INT,

    city TEXT,
    country TEXT,
    region TEXT,
    latitude FLOAT,
    longitude FLOAT,
    population BIGINT,

    PRIMARY KEY (latitude, longitude, localObsDateTime)
);

-- 2. Copia i dati dalla vecchia tabella alla nuova tabella
INSERT INTO weather_new (
    localObsDateTime, temp, feelsLike, tempF, feelsLikeF, cloudcover, humidity, uvIndex,
    precip, precipInches, pressure, pressureInches, visibility, visibilityMiles,
    weatherCode, weatherDescription, winddir, winddir16Point, windspeed, windspeedMiles,
    city, country, region, latitude, longitude, population
)
SELECT
    localObsDateTime, temp, feelsLike, tempF, feelsLikeF, cloudcover, humidty, uvIndex,
    precip, precipInches, pressure, pressureInches, visibility, visibilityMiles,
    weatherCode, weatherDescription, winddir, winddir16Point, windspeed, windspeedMiles,
    city, country, region, latitude, longitude, population
FROM weather;

-- 3. Elimina la vecchia tabella e rinomina la nuova
DROP TABLE weather;
ALTER TABLE weather_new RENAME TO weather;

-- Riattiva le chiavi esterne
PRAGMA foreign_keys = ON;
"""

REMOVE_QUOTES = """
UPDATE weather
SET region = NULL
WHERE region = "";
"""