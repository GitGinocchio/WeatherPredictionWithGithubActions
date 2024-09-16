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
    # Define the API URL for fetching weather data for a given city
    api_url = f'https://wttr.in/{city}?format=j1'
    
    try:
        # Send a GET request to the API and wait up to 60 seconds for a response
        response = session.get(api_url, timeout=60)
        
        # Assert that the response content type is JSON (application/json)
        assert response.headers['Content-Type'] == 'application/json', f"Expected Content-Type to be application/json, but got {response.headers['Content-Type']}"

        # Parse the JSON response into a Python dictionary
        current_report = response.json()

        # Extract the observation time from the current weather report
        current_obs_time = current_report["current_condition"][0]["localObsDateTime"]

        # TODO: (Maybe check only the latest report to avoid duplicates)
        # Loop through all files in the 'data/collected' directory
        for report in os.listdir('data/collected'): 
            # Check if a file with the same city name exists, but not the latest report (based on observation time)
            if not os.path.exists(f'data/collected/{report}/{city}.json'):
                continue

            with open(f'data/collected/{report}/{city}.json', 'r') as f:
                # Load the old weather report from a file
                old_report = json.load(f)

            # Extract the observation time from the old weather report
            old_obs_time = old_report["current_condition"][0]["localObsDateTime"]

            # If the observation times match, we can stop looking for newer reports
            if old_obs_time == current_obs_time: break
        else:
            # Create a new directory to store the fetched data for this timestamp
            data_path = os.path.join(r'data/collected', timestamp)
            os.makedirs(data_path, exist_ok=True)

            with open(os.path.join(data_path, f"{city}.json"), 'w') as f:
                # Write the current weather report to a new file in the created directory
                json.dump(current_report, f, indent=4)

    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.SSLError, AssertionError, json.JSONDecodeError) as e: 
        print(f'Error fetching {city} weather data:\n{e}')



# This is the main function of the script, which collects weather data for a list of cities and writes it to files.
def main(args : Namespace) -> None:
    # Get the current timestamp in the format YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S')

    # Create a ThreadPoolExecutor to run tasks concurrently (e.g., fetching weather data for multiple cities at once)
    with ThreadPoolExecutor(max_workers=len(config["sample-cities"])) as executor:
        # Define a list of tasks, where each task is a function call to fetch_city_weather_data for a given city and timestamp
        tasks = [executor.submit(fetch_city_weather_data, city, timestamp) for city in config["sample-cities"]]

    # Wait for all tasks to complete (i.e., retrieve the results)
    for task in tasks: task.result()

    # Add the timestamp to the list of collected timestamps if the timestamp is present in the file system
    if os.path.exists(f'data/collected/{timestamp}'):
        with open('data/collected/entities.txt','a') as entities:
            entities.write(f'{timestamp}\n')

# This is a special block of code that runs when this script is executed directly (i.e., not imported as a module)
if __name__ == '__main__':
    # Create an ArgumentParser object to parse command-line arguments
    parser = ArgumentParser()

    # Parse the command-line arguments and store them in the 'args' variable
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
