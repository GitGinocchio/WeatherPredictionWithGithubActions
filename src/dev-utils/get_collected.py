import requests
import json
import os

with open('config/sample-cities.json','r') as f:
    config = json.load(f)

def main() -> None:
    url = f'https://raw.githubusercontent.com/GitGinocchio/weather-prediction-with-github-actions/data/collected/entities.txt'
    response = requests.get(url)
    
    if response.status_code == 200:
       print('Successfully got entities.txt file')
       for directory in response.content.decode().split('\n'):
           print(f'Downloading directory: {directory}')
           os.makedirs(f'data/collected/{directory}',exist_ok=True)
           for city in config['sample-cities']:
                print(f'creating file: data/collected/{directory}/{city}.json')
                resource_url = f'https://raw.githubusercontent.com/GitGinocchio/weather-prediction-with-github-actions/data/collected/{directory}/{city}.json'
                response = requests.get(resource_url)
                with open(f'data/collected/{directory}/{city}.json','wb') as f:
                    f.write(response.content)
    else:
        raise Exception(f"Failed to get entities file: {response.status_code}")

main()

