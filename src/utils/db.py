from datetime import datetime, timezone
from typing import Generator, Any
import sqlite3
import uuid

from utils.queries import *

DATABASE_PATH = "./data/database.db"
SCRIPT_PATH = "./config/database.sql"

class Database:
    _instance = None
    def __init__(self):
        self._connection : sqlite3.Connection
        self._cursor : sqlite3.Cursor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._connection = sqlite3.connect(DATABASE_PATH)
            cls._connection.row_factory = sqlite3.Row

            with open(SCRIPT_PATH, "r") as f:
                cls._connection.executescript(f.read())
        return cls._instance

    def __enter__(self):
        self._cursor = self.connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self._cursor.close()
        self._cursor = None

    @property
    def connection(self) -> sqlite3.Connection:
        assert self._connection is not None, "Connection has not been created yet"
        return self._connection
    
    @property
    def cursor(self) -> sqlite3.Cursor:
        assert self._cursor is not None, "Cursor has not been created yet"
        return self._cursor
    
    def createBackup(self):
        pass

    def executeQueryScript(self, query : str) -> None:
        try:
            self.cursor.executescript(query)
        except sqlite3.IntegrityError as e:
            self.connection.rollback()
            raise e
        except Exception as e:
            self.connection.rollback()
            raise e
        else:
            self.connection.commit()

    def hasWeatherCondition(self, latitude : int, longitude : int, datetime : str) -> dict[str, Any] | None:
        cursor = self.cursor.execute(HAS_WEATHER_CONDITION, (latitude, longitude, datetime))
        return dict(row) if (row:=cursor.fetchone()) else None

    def hasDaily(self, latitude : int, longitude : int, date : str) -> dict[str, Any] | None:
        cursor = self.cursor.execute(HAS_DAILY, (latitude, longitude, date))
        return dict(row) if (row:=cursor.fetchone()) else None

    def getAllWeatherConditions(self) -> Generator[dict[str, Any], None, None]:
        cursor = self.cursor.execute(GET_ALL_WEATHER_CONDITIONS)
        for row in cursor.fetchall():
            yield dict(row)

    def newReport(self, report : dict[str, list[dict]]) -> None:
        try:
            condition = report["current_condition"][0]
            daily = report["weather"]
            area = report["nearest_area"][0]
            has_daily = False


            if (self.hasDaily(area["latitude"], area["longitude"], daily[0]["date"])) is not None:
                has_daily = True

            self.cursor.execute(NEW_WEATHER_CONDITION_QUERY, (
                condition["localObsDateTime"],
                condition["temp_C"],
                condition["FeelsLikeC"],
                
                condition["temp_F"],
                condition["FeelsLikeF"],
                
                condition["cloudcover"],
                condition["humidity"],
                condition["uvIndex"],
                
                condition["precipMM"],
                condition["precipInches"],
                
                condition["pressure"],
                condition["pressureInches"],

                condition["visibility"],
                condition["visibilityMiles"],

                condition["weatherCode"],
                condition["weatherDesc"][0]["value"],

                condition["winddirDegree"],
                condition["winddir16Point"],
                condition["windspeedKmph"],
                condition["windspeedMiles"],

                area["areaName"][0]["value"],
                area["country"][0]["value"],
                region if (region:=area["region"][0]["value"]) else None,

                area["latitude"],
                area["longitude"],

                area["population"]
            ))

            if has_daily:
                self.connection.commit()
                return

            for day in range(0, len(daily)):
                self.cursor.execute(NEW_DAILY_QUERY, (
                    daily[day]["date"],

                    area["latitude"],
                    area["longitude"],
                    
                    daily[day]["astronomy"][0]["moon_illumination"],
                    daily[day]["astronomy"][0]["moon_phase"],
                    
                    daily[day]["astronomy"][0]["moonrise"],
                    daily[day]["astronomy"][0]["moonset"],
                    
                    daily[day]["astronomy"][0]["sunrise"],
                    daily[day]["astronomy"][0]["sunset"],

                    daily[day]["avgtempC"],
                    daily[day]["avgtempF"],

                    daily[day]["maxtempC"],
                    daily[day]["maxtempF"],

                    daily[day]["mintempC"],
                    daily[day]["mintempF"],

                    daily[day]["sunHour"],
                    daily[day]["totalSnow_cm"],
                    daily[day]["uvIndex"],
                ))

                hourly = daily[day]["hourly"]

                for hour in range(0,len(daily[day]["hourly"])):
                    self.cursor.execute(NEW_HOURLY_QUERY,(
                        daily[day]["date"],
                        hourly[hour]["time"],

                        area["latitude"],
                        area["longitude"],

                        hourly[hour]["DewPointC"],
                        hourly[hour]["DewPointF"],
                        
                        hourly[hour]["FeelsLikeC"],
                        hourly[hour]["FeelsLikeF"],

                        hourly[hour]["HeatIndexC"],
                        hourly[hour]["HeatIndexF"],

                        hourly[hour]["WindChillC"],
                        hourly[hour]["WindChillF"],

                        hourly[hour]["WindGustKmph"],
                        hourly[hour]["WindGustMiles"],
                        
                        hourly[hour]["chanceoffog"],
                        hourly[hour]["chanceoffrost"],
                        hourly[hour]["chanceofhightemp"],
                        hourly[hour]["chanceofovercast"],
                        hourly[hour]["chanceofrain"],
                        hourly[hour]["chanceofremdry"],
                        hourly[hour]["chanceofsnow"],
                        hourly[hour]["chanceofsunshine"],
                        hourly[hour]["chanceofthunder"],
                        hourly[hour]["chanceofwindy"],

                        hourly[hour]["shortRad"],
                        hourly[hour]["diffRad"],

                        hourly[hour]["cloudcover"],
                        hourly[hour]["humidity"],
                        hourly[hour]["uvIndex"],

                        hourly[hour]["precipMM"],
                        hourly[hour]["precipInches"],

                        hourly[hour]["pressure"],
                        hourly[hour]["pressureInches"],

                        hourly[hour]["tempC"],
                        hourly[hour]["tempF"],

                        hourly[hour]["visibility"],
                        hourly[hour]["visibilityMiles"],

                        hourly[hour]["weatherCode"],
                        hourly[hour]["weatherDesc"][0]["value"],
                        
                        hourly[hour]["windspeedKmph"],
                        hourly[hour]["windspeedMiles"],

                        hourly[hour]["winddirDegree"],
                        hourly[hour]["winddir16Point"],
                    ))

            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e
        except sqlite3.IntegrityError as e:
            self.connection.rollback()
            raise e

    def updateInfo(self):
        pass