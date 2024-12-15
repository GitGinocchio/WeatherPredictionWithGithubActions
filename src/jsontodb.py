from utils.db import Database
import sqlite3
import json
import os


db = Database()


with db as conn:
    for report in os.listdir("./data/reports"):
        for city in os.listdir(f"./data/reports/{report}"):
            with open(f"./data/reports/{report}/{city}") as f:
                data = json.load(f, parse_int=True, parse_float=True)

                latitude = float(data["nearest_area"][0]["latitude"])
                longitude = float(data["nearest_area"][0]["longitude"])
                datetime = str(data["current_condition"][0]["localObsDateTime"])

                if conn.hasWeatherCondition(latitude, longitude, datetime):
                    print(f"report already exists for city: {city} and report: {report}")
                    continue

                print(f"creating report {report} for {city}")


                try:
                    conn.newReport(data)
                except sqlite3.IntegrityError as e:
                    print(e)