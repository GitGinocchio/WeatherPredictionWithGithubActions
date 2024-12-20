
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