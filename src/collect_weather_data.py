from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from tqdm import tqdm
import requests
import asyncio
import json
import sys
import os

from utils.db import Database
from utils.terminal import getlogger
from utils.config import config

logger = getlogger()
db = Database()

session = requests.Session()

def fetch_city_weather_data(city : str) -> dict | None:
    try:
        #logger.info(f"Fetching weather data for city: {city}")
        response = session.get(f'https://wttr.in/{city}?format=j1', timeout=10, allow_redirects=False)
        
        assert response.headers['Content-Type'] == 'application/json', f"Expected Content-Type to be application/json, but got {response.headers['Content-Type']}"

        report : dict = response.json()

    except (requests.exceptions.ConnectTimeout, 
            requests.exceptions.ReadTimeout, 
            requests.exceptions.SSLError, 
            json.JSONDecodeError, 
            requests.exceptions.ConnectionError,
            AssertionError) as e:
        logger.error(e)
        return None
    else:
        return report

def main(args : Namespace) -> None:

    logging_queue : list[str] = []

    with db as conn:
        with tqdm(config["sample-cities"], desc="Fetching weather data", unit="cities", total=len(config["sample-cities"])) as bar:
            for city in config["sample-cities"]:
                report = fetch_city_weather_data(city)

                bar.update()

                if not report: continue

                latitude = report["nearest_area"][0]["latitude"]
                longitude = report["nearest_area"][0]["longitude"]
                dt = datetime.strptime(report["current_condition"][0]["localObsDateTime"], "%Y-%m-%d %I:%M %p")

                if conn.hasWeatherCondition(latitude, longitude, dt.year, dt.month, dt.day, dt.hour, dt.minute):
                    logging_queue.append(f"Report for {city:<15} at {dt} already exists. Skipping.")
                    continue

                conn.newReport(report)
                logging_queue.append(f"Report for {city:<15} at {dt} created successfully.")
        
        logger.info(str(bar))
        for log in logging_queue:
            logger.info(log)


if __name__ == '__main__':
    # Create an ArgumentParser object to parse command-line arguments
    parser = ArgumentParser()

    # Parse the command-line arguments and store them in the 'args' variable
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
