BEGIN;

CREATE TABLE IF NOT EXISTS info (
    lastUpdate DATETIME PRIMARY KEY,
    numReports BIGINT,
    numDailies BIGINT,
    numHourly BIGINT
);

CREATE TABLE IF NOT EXISTS weather (
    localObsDateTime DATETIME,

    temp INT,
    feelsLike INT,

    tempF INT,
    feelsLikeF INT,

    cloudcover INT,
    humidty INT,
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

CREATE TABLE IF NOT EXISTS daily (
    localObsDateTime DATETIME,
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

    PRIMARY KEY (date, latitude, longitude)
    FOREIGN KEY (localObsDateTime, latitude, longitude) REFERENCES weather (localObsDateTime, latitude, longitude)
);

CREATE TABLE IF NOT EXISTS hourly (
    localObsDateTime DATETIME,
    date DATETIME,
    time INT,                           -- vengono fatte 8 rilevazione rispettivamente a: 0, 3, 6, 9, 12, 15, 18 e 21.

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

    PRIMARY KEY (date, time, latitude, longitude)
    FOREIGN KEY (date, latitude, longitude) REFERENCES daily (date, latitude, longitude)
    FOREIGN KEY (localObsDateTime, latitude, longitude) REFERENCES weather (localObsDateTime, latitude, longitude)
);

COMMIT;