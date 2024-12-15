import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from utils.dataloader import stream_data_in_memory

# Caricamento e pre-elaborazione dei dati
data = stream_data_in_memory()
df = pd.DataFrame(data)

def try_convert_numeric(val):
    try:
        return pd.to_numeric(val)
    except (ValueError, TypeError):
        return val

df = df.map(try_convert_numeric)

label_encoder = LabelEncoder()
categorical_columns = ['area', 'region', 'country', 'winddir16Point', 'weatherDesc']
for col in categorical_columns:
    df[f'{col}_encoded'] = label_encoder.fit_transform(df[col])

# Selezione delle feature e target
X = df[['minute', 'hour', 'month', 'day', 'year', 'latitude', 'longitude']] # 'area_encoded', 'region_encoded', 'country_encoded'
y = df[["FeelsLikeC", "cloudcover", "humidity", "precipMM", "pressure", "temp_C", "uvIndex", "visibility", "windspeedKmph"]]

# Divisione dei dati e normalizzazione
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# Conversione in tensori PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to("cuda")
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to("cuda")
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to("cuda")

# Definizione del modello
class ImprovedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.batch_norm1(self.layer1(x))))
        x = self.dropout(torch.relu(self.batch_norm2(self.layer2(x))))
        x = self.dropout(torch.relu(self.batch_norm3(self.layer3(x))))
        x = self.output(x)
        return x

# Preparazione del dataset e dataloader
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Inizializzazione del modello e dell'ottimizzatore
input_dim = X_train_tensor.shape[1]
hidden_dim = 128
output_dim = y_train_tensor.shape[1]

model = ImprovedNN(input_dim, hidden_dim, output_dim).to("cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# Training loop
num_epochs = 10000
best_val_loss = float('inf')
patience = 30
no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            y_val_pred = model(X_val)
            val_loss += criterion(y_val_pred, y_val).item()
    val_loss /= len(val_loader)
    
    scheduler.step(val_loss)
    
    print(f'Epoca {epoch+1}/{num_epochs}, Loss di training: {train_loss:.5f}, Loss di validazione: {val_loss:.5f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print("Early stopping")
        break

# Caricamento del miglior modello
model.load_state_dict(torch.load('best_model.pth'))

# Funzione per fare previsioni
def predict(model, data):
    model.eval()
    with torch.no_grad():
        scaled_data = scaler_X.transform(data)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32).to("cuda")
        predictions = model(tensor_data)
        return scaler_y.inverse_transform(predictions.cpu().numpy())

# Esempio di previsione
new_data = pd.DataFrame({
    'minute': [0],
    'hour': [13],
    'month': [9],
    'day': [15],
    'year': [2024],
    'latitude': [45.549999],
    'longitude': [11.550000],
    #'area_encoded': [3],
    #'region_encoded': [9],
    #'country_encoded': [7]
})

result = predict(model, new_data)
columns = [
    'FeelsLikeC', 'cloudcover', 'humidity', 'precipMM',
    'pressure', 'temp_C', 'uvIndex', 'visibility', 'windspeedKmph'
]

print(pd.DataFrame(result, columns=columns))

# Valutazione del modello sul set di test
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, torch.tensor(scaler_y.transform(y_test), dtype=torch.float32).to("cuda"))
    print(f'Loss sul set di test: {test_loss.item():.5f}')

# Calcolo dell'R-squared per ogni variabile target
y_pred_np = scaler_y.inverse_transform(y_pred.cpu().numpy())
y_test_np = y_test.values
r2_scores = {}

for i, col in enumerate(y.columns):
    r2 = 1 - np.sum((y_test_np[:, i] - y_pred_np[:, i])**2) / np.sum((y_test_np[:, i] - np.mean(y_test_np[:, i]))**2)
    r2_scores[col] = r2

print("\nR-squared per ogni variabile target:")
for col, r2 in r2_scores.items():
    print(f"{col}: {r2:.4f}")