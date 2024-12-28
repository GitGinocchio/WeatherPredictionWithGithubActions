
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


    minTemp INT,
    minTempF INT,

    sunHour FLOAT,
    totalSnow FLOAT,
    uvIndex INT,

    PRIMARY KEY (year, month, day, latitude, longitude),
    FOREIGN KEY (latitude, longitude, year, month, day) REFERENCES weather (latitude, longitude, year, month, day)
);
"""

