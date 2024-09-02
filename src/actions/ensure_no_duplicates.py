import json
import os

with open(r"config/sample-cities.json",'r') as f:
    config = json.load(f)

def main() -> None:
    seen_dates = set()
    for city in config['sample-cities']:
        for report in os.listdir('data/collected'):
            if report == 'entities.txt': continue

            if not os.path.exists(f'data/collected/{report}/{city}.json'):
                if len(os.listdir(f'data/collected/{report}')) == 0:
                    os.removedirs(f'data/collected/{report}')
                continue

            with open(f'data/collected/{report}/{city}.json','r') as f:
                content = json.load(f)

            local_obs_time = content["current_condition"][0]["localObsDateTime"]

            if local_obs_time in seen_dates:
                os.remove(f'data/collected/{report}/{city}.json')
            else:
                seen_dates.add(local_obs_time)

            if len(os.listdir(f'data/collected/{report}')) == 0:
                os.removedirs(f'data/collected/{report}')
        seen_dates.clear()

    with open(f'data/collected/entities.txt', 'w') as f:
        for report in os.listdir('data/collected'):
            if report != 'entities.txt': f.write(report+'\n')

if __name__ == '__main__':
    main()