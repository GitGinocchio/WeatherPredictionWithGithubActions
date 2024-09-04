from argparse import ArgumentParser, Namespace
import datetime
import requests
import json
import os

with open(r"config/sample-cities.json",'r') as f:
    config = json.load(f)

def fetch_weather_data(city: str) -> dict:
    # Funzione per ottenere i dati meteo per una città
    api_url = f'https://wttr.in/{city}?format=j1'
    try:
        response = requests.get(api_url,timeout=1)
        if response.apparent_encoding == 'ascii': return None
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
        print(f'Error fetching {city} weather data:')
        print(e)
        return None
    else:
        return response.json()

def save_weather_data(data : dict, timestamp : str) -> None:
    # Crea una cartella per l'orario se non esiste già
    data_path = os.path.join(r'data/collected', timestamp)
    os.makedirs(data_path, exist_ok=True)

    # Salva i dati per ogni città in un file separato
    for city, city_data in data.items():
        file_path = os.path.join(data_path, f"{city}.json")
        with open(file_path, 'w') as f:
            json.dump(city_data, f, indent=4)

    with open('data/collected/entities.txt','a') as entities:
        entities.write(f'\n{timestamp}')

def main(args : Namespace) -> None:
    timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S')

    seen_dates = set()
    weather_data = {}
    for city in config['sample-cities']:
        content = fetch_weather_data(city)

        if not content: continue

        for report in os.listdir('data/collected'):
            if not os.path.exists(f'data/collected/{report}/{city}.json'):
                continue

            local_obs_time = content["current_condition"][0]["localObsDateTime"]

            if local_obs_time in seen_dates: break
        else:
            weather_data[city] = content
        seen_dates.clear()
    
    if len(weather_data) == 0: return

    save_weather_data(weather_data,timestamp)





if __name__ == '__main__':
    parser = ArgumentParser()

    args = parser.parse_args()

    main(args)