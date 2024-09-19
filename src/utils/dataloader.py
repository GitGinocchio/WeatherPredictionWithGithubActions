from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
import io
import requests
import json
import os

CACHE_PATH = '.cache/dataloader.cache'
session = requests.Session()

with open('config/sample-cities.json','r') as f:
	config = json.load(f)

def get_city_data(report : str, city : str) -> tuple[str, str, dict | Exception]:
	resource_url = f'https://raw.githubusercontent.com/GitGinocchio/weather-prediction-with-github-actions/data/reports/{report}/{city}.json'
	
	try:
		response = session.get(resource_url)

		assert response.status_code != 404, f'No data found for \'{city}\' in report: {report}'
		assert response.status_code == 200, f'Could not fetch \'{city}\' weather data for report: {report}'
	except AssertionError as e:
		return (report, city, e)
	except requests.exceptions.ConnectionError as e:
		return (report, city, e)
	else:
		return (report, city, response.json(parse_float=True,parse_int=True))

def stream_data_in_memory(enable_cache : bool = True):
	url = f'https://raw.githubusercontent.com/GitGinocchio/weather-prediction-with-github-actions/data/entities.json'

	try:
		response = session.get(url)

		assert response.status_code == 200, f"Failed to get entities file: {response.status_code}"
		entities = json.loads(response.content)
		last_update = entities['last-update']

		print(f'Successfully got entities.json file. Last available update: {last_update}')

		reports = entities['reports'].items()

		with ThreadPoolExecutor(max_workers=len(reports) * len(config['sample-cities'])) as executor:
			futures = [
				executor.submit(get_city_data, report, city) 
				for report,cities in tqdm(reports,desc='Downloading weather reports',unit='report') 
				for city in config['sample-cities']
				if city in cities
			]

			for future in as_completed(futures):
				report, city, result = future.result()

				if type(result) != dict:
					print(f'Failed to get data for "{city}" in report: "{report}": {result}')
					continue

				useful_data = result['current_condition'][0]
				useful_data['area'] = result['nearest_area'][0]['areaName'][0]['value']
				useful_data['country'] = result['nearest_area'][0]['country'][0]['value']
				useful_data['latitude'] = result['nearest_area'][0]['latitude']
				useful_data['longitude'] = result['nearest_area'][0]['longitude']
				useful_data['population'] = result['nearest_area'][0]['population']
				useful_data['region'] = result['nearest_area'][0]['region'][0]['value']
				useful_data['weatherDesc'] = result['current_condition'][0]['weatherDesc'][0]['value']

				useful_data['localObsDateTime'] = datetime.strptime(useful_data['localObsDateTime'], '%Y-%m-%d %I:%M %p')
				useful_data['minute'] = useful_data['localObsDateTime'].minute
				useful_data['hour'] = useful_data['localObsDateTime'].hour
				useful_data['day'] = useful_data['localObsDateTime'].day
				useful_data['month'] = useful_data['localObsDateTime'].month
				useful_data['year'] = useful_data['localObsDateTime'].year

				useful_data.pop("observation_time")
				useful_data.pop("weatherIconUrl")

				yield dict(useful_data)
			else:
				print("Successfully loaded all available data...")
	except (AssertionError,requests.exceptions.ConnectionError) as e:
		print(e)

for item in stream_data_in_memory():
	pass