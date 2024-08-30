import os
import json
import pandas as pd
from datetime import datetime

with open('config/sample-cities.json','r') as f:
    config = json.load(f)

def fetch_unique_data(directory : str):
    """
    Upload all unique data checking for each city the date on which the weather conditions were detected thus avoiding data duplication
    

    returns:\n
    unique_files -> list:
        The number of unique files found
    num_total_files -> int:
        The total number of files fetched

    return (unique_files, num_total_files)
    """

    seen_dates = set()
    num_unique_files = 0
    reports = []

    for city in (sample_cities:=config['sample-cities']):
        for report in (reports:=os.listdir(directory)):
            with open(f'{directory}/{report}/{city}.json', 'r') as f:
                content = json.load(f)

                local_obs_time = content["current_condition"][0]["localObsDateTime"]

                if local_obs_time not in seen_dates:
                    seen_dates.add(local_obs_time)
                    num_unique_files+=1

                    """
                    {
                        "FeelsLikeC": "13",
                        "FeelsLikeF": "56",
                        "cloudcover": "100",
                        "humidity": "89",
                        "localObsDateTime": "2024-08-28 04:00 PM",
                        "observation_time": "12:00 AM",
                        "precipInches": "0.0",
                        "precipMM": "0.4",
                        "pressure": "1007",
                        "pressureInches": "30",
                        "temp_C": "13",
                        "temp_F": "55",
                        "uvIndex": "3",
                        "visibility": "16",
                        "visibilityMiles": "9",
                        "weatherCode": "296",
                        "weatherDesc": "Light rain",
                        "winddir16Point": "NNW",
                        "winddirDegree": "340",
                        "windspeedKmph": "9",
                        "windspeedMiles": "6"
                        "area_name": "Anchorage",
                        "country": "United States of America"
                        "latitude": "61.218",
                        "longitude": "-149.900",
                        "population": "276263",
                        "region": "Alaska"
                    }
                    """

                    useful_data = content['current_condition'][0]
                    useful_data['area'] = content['nearest_area'][0]['areaName'][0]['value']
                    useful_data['country'] = content['nearest_area'][0]['country'][0]['value']
                    useful_data['latitude'] = content['nearest_area'][0]['latitude']
                    useful_data['longitude'] = content['nearest_area'][0]['longitude']
                    useful_data['population'] = content['nearest_area'][0]['population']
                    useful_data['region'] = content['nearest_area'][0]['region'][0]['value']
                    useful_data['weatherDesc'] = content['current_condition'][0]['weatherDesc'][0]['value']

                    useful_data['localObsDateTime'] = datetime.strptime(useful_data['localObsDateTime'], '%Y-%m-%d %I:%M %p')
                    useful_data['hour'] = useful_data['localObsDateTime'].hour
                    useful_data['day'] = useful_data['localObsDateTime'].day
                    useful_data['month'] = useful_data['localObsDateTime'].month
                    useful_data['year'] = useful_data['localObsDateTime'].year

                    useful_data.pop("observation_time")
                    useful_data.pop("weatherIconUrl")

                    yield useful_data

        seen_dates = set()
    return num_unique_files, len(sample_cities) * len(reports) if reports else 0
