from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
import requests
import asyncio
import json
import os

from utils.db import Database

db = Database()

session = requests.Session()

with open(r"config/sample-cities.json",'r') as f:
    config = json.load(f)

def fetch_city_weather_data(city : str) -> dict | None:
    api_url = f'https://wttr.in/{city}?format=j1'
    
    try:
        response = session.get(api_url, timeout=60)
        
        assert response.headers['Content-Type'] == 'application/json', f"Expected Content-Type to be application/json, but got {response.headers['Content-Type']}"

        report : dict = response.json()

    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.SSLError, json.JSONDecodeError) as e: 
        print(f'Error fetching {city} weather data:\n{e}')
        return None
    except AssertionError as e:
        print(e)
        return None
    else:
        return report

def main(args : Namespace) -> None:
    with db as conn:
        for city in config["sample-cities"]:
            report = fetch_city_weather_data(city)

            if not report: continue

            latitude = report["nearest_area"][0]["latitude"]
            longitude = report["nearest_area"][0]["longitude"]
            datetime = report["current_condition"][0]["localObsDateTime"]

            if conn.hasWeatherCondition(latitude, longitude, datetime):
                continue

            conn.newReport(report)



if __name__ == '__main__':
    # Create an ArgumentParser object to parse command-line arguments
    parser = ArgumentParser()

    # Parse the command-line arguments and store them in the 'args' variable
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
