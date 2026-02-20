import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, features: pd.DataFrame, target: pd.Series, sequence_length: int = 12):
        self.features = features.values
        self.target = target.values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.sequence_length]
        y = self.target[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class NeuralSequenceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2):
        super(NeuralSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze(-1)


def train_neural_model(
    model: NeuralSequenceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    device: str = "cpu",
):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_preds = model(x_val)
                val_loss = criterion(val_preds, y_val)
                val_losses.append(val_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_neural_model.pth")

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(torch.load("best_neural_model.pth"))
    return model


def neural_inference(model: NeuralSequenceModel, features: np.ndarray, sequence_length: int = 12, device: str = "cpu") -> np.ndarray:
    model.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(len(features) - sequence_length):
            seq = features[i : i + sequence_length]
            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(seq_t).item()
            preds.append(pred)

    preds_np = np.array(preds)
    preds_clipped = np.clip(preds_np * 2 - 1, -1, 1)  # Scale to [-1, 1]
    return preds_clipped
