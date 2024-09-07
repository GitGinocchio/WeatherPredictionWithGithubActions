from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
import datetime
import requests
import asyncio
import json
import os

session = requests.Session()

with open(r"config/sample-cities.json",'r') as f:
    config = json.load(f)

def fetch_city_weather_data(city : str, timestamp : str):
    api_url = f'https://wttr.in/{city}?format=j1'

    try:
        response = session.get(api_url, timeout=60)
        assert response.headers['Content-Type'] == 'application/json', f"Expected Content-Type to be application/json, but got {response.headers['Content-Type']}"
        
        current_report = response.json()

        current_obs_time = current_report["current_condition"][0]["localObsDateTime"]

        for report in os.listdir('data/collected'): # Controllo ogni report
            if not os.path.exists(f'data/collected/{report}/{city}.json'):
                continue

            with open(f'data/collected/{report}/{city}.json', 'r') as f:
                old_report = json.load(f)

            old_obs_time = old_report["current_condition"][0]["localObsDateTime"]

            if old_obs_time == current_obs_time: break
        else:
            data_path = os.path.join(r'data/collected', timestamp)
            os.makedirs(data_path, exist_ok=True)
            with open(os.path.join(data_path, f"{city}.json"), 'w') as f:
                json.dump(current_report, f, indent=4)

    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, AssertionError, json.JSONDecodeError) as e: 
        print(f'Error fetching {city} weather data:\n{e}')



def main(args : Namespace) -> None:
    timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S')

    with ThreadPoolExecutor() as executor:
        tasks = [executor.submit(fetch_city_weather_data, city, timestamp) for city in config["sample-cities"]]
    
    for task in tasks: task.result()

    with open('data/collected/entities.txt','w') as entities:
        for report in os.listdir('data/collected'):
            if report == 'entities.txt': continue
            entities.write(f'{report}\n')


if __name__ == '__main__':
    parser = ArgumentParser()

    args = parser.parse_args()
    
    main(args)