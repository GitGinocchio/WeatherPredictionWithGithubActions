from argparse import ArgumentParser, Namespace
import datetime
import requests
import json
import os

with open(r"config\sample-cities.json",'r') as f:
    config = json.load(f)

def fetch_weather_data(city: str) -> dict:
    # Funzione per ottenere i dati meteo per una città
    api_url = f'https://wttr.in/{city}?format=j1'
    response = requests.get(api_url)
    return response.json()

def save_weather_data(data : dict, timestamp : str) -> None:
    # Crea una cartella per l'orario se non esiste già
    folder_path = os.path.join(r'data\collected', timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # Salva i dati per ogni città in un file separato
    for city, city_data in data.items():
        file_path = os.path.join(folder_path, f"{city}.json")
        with open(file_path, 'w') as f:
            json.dump(city_data, f, indent=4)

def main(args : Namespace) -> None:
    timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S')

    weather_data = {}
    for city in config['sample-cities']:
        weather_data[city] = fetch_weather_data(city)

    save_weather_data(weather_data,timestamp)





if __name__ == '__main__':
    parser = ArgumentParser()

    args = parser.parse_args()

    main(args)