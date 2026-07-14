import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from core.base_model import BaseGenerativeModel
from core.pipeline import DataPipeline
from typing import Dict, Any

class PyTorchAutoencoder(nn.Module):
    """Internal PyTorch layout handling compression/expansion math layers."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class DeepGenerativeModel(BaseGenerativeModel):
    """Concrete PyTorch generative pipeline integrating structural DataPipeline objects."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.latent_dim = config.get("latent_dim", 2)
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        self.lr = config.get("lr", 0.01)
        
        self.nn_model = None
        self.pipeline = DataPipeline()  # Composition design pattern

    def fit(self, data: pd.DataFrame) -> None:
        print("[INFO] Initializing Deep Learning Pipeline...")
        input_dim = data.shape[1]
        
        # Isolate scaling tasks to core pipeline component
        scaled_data = self.pipeline.fit_transform(data)
        
        tensor_x = torch.tensor(scaled_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.nn_model = PyTorchAutoencoder(input_dim, self.latent_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=self.lr)

        self.nn_model.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                inputs = batch[0]
                outputs = self.nn_model(inputs)
                loss = criterion(outputs, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        self.is_trained = True
        print("[INFO] Deep Learning Model training complete.")

    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating data.")
            
        self.nn_model.eval()
        with torch.no_grad():
            noise = torch.randn(num_rows, self.latent_dim)
            synthetic_scaled = self.nn_model.decoder(noise).numpy()
            
        # Format metrics back into original column distributions and layouts cleanly
        return self.pipeline.inverse_transform(synthetic_scaled)