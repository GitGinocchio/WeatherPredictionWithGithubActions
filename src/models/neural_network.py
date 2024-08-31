from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



import pandas as pd
import numpy as np

from utils.preprocessing import fetch_unique_data



data = fetch_unique_data('data/collected')
df = pd.DataFrame(data)

def try_convert_numeric(val):
    try:
        return pd.to_numeric(val)
    except (ValueError, TypeError):
        return val

df = df.map(try_convert_numeric)

label_encoder = LabelEncoder()

df['area_encoded'] = label_encoder.fit_transform(df['area'])
df['region_encoded'] = label_encoder.fit_transform(df['region'])
df['country_encoded'] = label_encoder.fit_transform(df['country'])
df['winddir16Point_encoded'] = label_encoder.fit_transform(df['winddir16Point'])
df['weatherDesc_encoded'] = label_encoder.fit_transform(df['weatherDesc'])

#print(df[['area','area_encoded','region','region_encoded','country','country_encoded','latitude','longitude','population']].tail(50))

# Inputs: giorno, ora, mese, anno, area, country, region
# Outputs: tutto il resto

X = df[['minute','hour','month','day','year','latitude','longitude']] #'area_encoded','region_encoded','country_encoded','population'
y = df[["FeelsLikeC","cloudcover","humidity","precipMM","pressure","temp_C","uvIndex","visibility","windspeedKmph"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25)

#base_model = RandomForestRegressor()
#model = MultiOutputRegressor(base_model)

#model = LinearRegression()

#model = RandomForestRegressor()

#poly = PolynomialFeatures(degree=2)
#model = make_pipeline(poly, LinearRegression())

# torch start
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x
    
# Converti i DataFrame in array NumPy
X_train_np = X_train.apply(pd.to_numeric, errors='coerce')
y_train_np = y_train.apply(pd.to_numeric, errors='coerce')
X_test_np = X_test.apply(pd.to_numeric, errors='coerce')

X_train_np = X_train_np.to_numpy()  # o X_train.values
y_train_np = y_train_np.to_numpy()
X_test_np = X_test_np.to_numpy()

# Converti gli array NumPy in tensori PyTorch
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to("cuda")
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to("cuda")
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to("cuda")

# Creazione del dataset e suddivisione in train/validation
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

input_dim = X_train_tensor.shape[1]
output_dim = y_train_tensor.shape[1]

model = SimpleNN(input_dim, output_dim).to("cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eseguiamo il training per 100 epoche
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()  # Settiamo il modello in modalità training
    for X_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Valutazione sul validation set
    model.eval()  # Settiamo il modello in modalità evaluation
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            y_val_pred = model(X_val)
            val_loss += criterion(y_val_pred, y_val).item()
    val_loss /= len(val_loader)
    
    # Stampa della loss per ogni epoca
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.5f}, Validation Loss: {val_loss:.5f}')

new_data = pd.DataFrame({
    #'area_encoded': [3],  # Sostituisci con i tuoi dati reali
    #'region_encoded': [9],
    #'country_encoded': [7],
    'minute' : [24],
    'hour': [17],
    'month': [8],
    'day': [31],
    'year': [2024],
    #'population': [3000],
    'latitude': [45.467],
    'longitude': [9.200]
})

result = model(torch.tensor(new_data.to_numpy(), dtype=torch.float32).to("cuda"))
columns = [
    'FeelsLikeC', 'cloudcover', 'humidity', 'precipMM',
    'pressure', 'temp_C', 'uvIndex', 'visibility', 'windspeedKmph'
]

print(pd.DataFrame(result.cpu().detach().numpy(),columns=columns))