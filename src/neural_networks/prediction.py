from neural_networks.models.lstm import LSTM
import numpy as np
import torch


def predict(model : LSTM, scaled_data : np.ndarray, model_path : str = 'model.pth'):
    tensors = torch.load(model_path,weights_only=True)
    model.load_state_dict(tensors)

    model.eval()
    with torch.no_grad():
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(1).to("cuda")
        predictions = model(tensor_data)
        return predictions.cpu().numpy()