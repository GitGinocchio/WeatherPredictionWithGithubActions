
NEW_WEATHER_CONDITION_QUERY = """
INSERT INTO weather (
    year,
    month,
    day,
    hour,
    minute,
    
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
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

NEW_DAILY_QUERY = """
INSERT INTO daily (
    year,
    month,
    day,
    
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
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

NEW_HOURLY_QUERY = """
INSERT INTO hourly (
    year,
    month,
    day,
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
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

HAS_WEATHER_CONDITION = """
SELECT * FROM weather 
WHERE latitude = ? 
  AND longitude = ? 
  AND year = ?
  AND month = ?
  AND day = ?
  AND hour = ?
  AND minute = ?
"""

GET_ALL_WEATHER_CONDITIONS = """
SELECT * FROM weather
"""

GET_ALL_DAILY = """
SELECT * FROM daily
"""

GET_ALL_HOURLY = """
SELECT * FROM hourly
"""

HAS_DAILY = """
SELECT * FROM daily 
WHERE latitude = ? 
  AND longitude = ? 
  AND year = ?
  AND month = ?
  AND day = ?
"""

CREATE_NEW_WEATHER_TABLE = """
CREATE TABLE IF NOT EXISTS weather_new (
    year INT, 
    month INT, 
    day INT, 
    hour INT, 
    minute INT,

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

    PRIMARY KEY (latitude, longitude, year, month, day, hour, minute)
);
"""

CREATE_NEW_DAILY_TABLE = """
CREATE TABLE IF NOT EXISTS daily_new (
    year INT, 
    month INT, 
    day INT, 

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

    PRIMARY KEY (year, month, day, latitude, longitude),
    FOREIGN KEY (latitude, longitude, year, month, day) REFERENCES weather (latitude, longitude, year, month, day)
);
"""

CREATE_NEW_HOURLY_TABLE = """
CREATE TABLE IF NOT EXISTS hourly_new (
    year INT, 
    month INT, 
    day INT,
    time INT,

    latitude FLOAT,
    longitude FLOAT,

    dewPoint INT,
    dewPointF INT,

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

    PRIMARY KEY (year, month, day, time, latitude, longitude),
    FOREIGN KEY (year, month, day, latitude, longitude) REFERENCES daily (year, month, day, latitude, longitude),
    FOREIGN KEY (year, month, day, latitude, longitude) REFERENCES weather (year, month, day, latitude, longitude)
);
"""
