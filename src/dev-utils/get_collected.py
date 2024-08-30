from concurrent.futures import ThreadPoolExecutor
import requests
import json
import os

with open('config/sample-cities.json','r') as f:
	config = json.load(f)

def download_city_data(report : str, city : str):
    resource_url = f'https://raw.githubusercontent.com/GitGinocchio/weather-prediction-with-github-actions/data/collected/{report}/{city}.json'
    response = requests.get(resource_url)
    if response.status_code == 200:
        file_path = f'data/collected/{report}/{city}.json'
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f'Successfully downloaded: {file_path}')
    else:
        print(f'Failed to download {city}.json for {report}: {response.status_code}')

def main() -> None:
	url = f'https://raw.githubusercontent.com/GitGinocchio/weather-prediction-with-github-actions/data/collected/entities.txt'
	response = requests.get(url)

	try:
		assert response.status_code == 200, f"Failed to get entities file: {response.status_code}"
		print('Successfully got entities.txt file')

		for report in response.content.decode().split('\n'):
			try: os.makedirs(f'data/collected/{report}')
			except OSError: continue
			else: print(f'Downloading report: {report}')

			with ThreadPoolExecutor() as executor:
				futures = [executor.submit(download_city_data, report, city) for city in config['sample-cities']]
				for future in futures:
					future.result()  # Assicurati che eventuali eccezioni vengano sollevate

	except AssertionError as e:
		raise AssertionError(f'AssertionError: {e}')
	except Exception as e:
		raise Exception(f'Failed to get entities: {e}')

main()