import datetime
import json
import os

entities = {
    "last-update" : datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S'),
    "num-reports" : len(os.listdir('reports')),
    "reports" : {

    }
}

for report in os.listdir('reports'):
    for city in os.listdir(f'reports/{report}'):
        if entities['reports'].get(report):
            entities['reports'][report].append(city.replace('.json',''))
        else:
            entities['reports'][report] = [city.replace('.json','')]

print(entities)

with open('entities.json','w') as f:
    json.dump(entities,f,indent=3)