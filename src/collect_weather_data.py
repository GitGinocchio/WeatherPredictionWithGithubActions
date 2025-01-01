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
from utils.terminal import getlogger, Level

logger = getlogger()
db = Database()

session = requests.Session()

def fetch_city_weather_data(city : str) -> dict | None:
    try:
        logger.debug(f"Fetching weather data for city: {city}")
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
    if args.cities:
        cities : list[str] = args.cities
    elif (environ_cities:=os.environ.get("CITIES", None)):
        cities = [city.strip() for city in environ_cities.split(",")]
    else:
        logger.error("No cities were specified. Please set the --cities flag or CITIES environment variable")
        sys.exit(1)

    logger.info(f"Starting to fetch weather data for cities: {", ".join(cities)}")
    with db as conn:
        for city in cities:
            report = fetch_city_weather_data(city)

            if not report: continue

            latitude : float = report["nearest_area"][0]["latitude"]
            longitude : float = report["nearest_area"][0]["longitude"]
            dt = datetime.strptime(report["current_condition"][0]["localObsDateTime"], "%Y-%m-%d %I:%M %p")

            if conn.hasWeatherCondition(latitude, longitude, dt.year, dt.month, dt.day, dt.hour, dt.minute):
                logger.info(f"Report for {city:<15} at {dt} already exists. Skipping.")
                continue

            conn.newReport(report)
            logger.info(f"Report for {city:<15} at {dt} created successfully.")

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Collect weather data for specified cities",
        epilog="For more information, visit https://github.com/GitGinocchio/WeatherPredictionWithGithubActions",
        add_help=True,
    )

    parser.add_argument(
        '-c', '--cities',
        nargs="*",
        type=lambda str: str.strip().replace("_"," "),
        required=True,
        default=None,
        help='Cities for which to collect weather data. Must be passed as a comma-separated, underscore spaced list of cities, or as an environment variable named CITIES.'
    )

    parser.add_argument(
        '-ll', '--log-level',
        type=str,
        help="Set the log level for the logger",
        default="INFO"
    )

    args = parser.parse_args()

    logger.setLevel(Level[args.log_level].value[0])

    main(args)
