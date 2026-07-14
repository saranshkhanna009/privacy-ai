import pandas as pd
import numpy as np

class DataPipeline:
    """
    Encapsulates all data preprocessing, scaling, and inverse transformations
    to keep raw mathematical transformations isolated from core AI loops.
    """
    
    def __init__(self):
        self.column_means = None
        self.column_stds = None
        self.feature_names = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Learns feature bounds and outputs scaled array for stable network processing."""
        print("[PIPELINE] Fitting transformations and scaling data...")
        self.feature_names = df.columns.tolist()
        self.column_means = df.mean().values
        self.column_stds = df.std().values
        
        # Prevent division by zero errors if feature variance is zero
        scaled_matrix = (df.values - self.column_means) / (self.column_stds + 1e-8)
        return scaled_matrix

    def inverse_transform(self, scaled_matrix: np.ndarray) -> pd.DataFrame:
        """Transforms processed matrices back into clean, real-world valued DataFrames."""
        if self.column_means is None:
            raise RuntimeError("Pipeline must be fitted before running inverse transformations.")
            
        real_scale_matrix = (scaled_matrix * (self.column_stds + 1e-8)) + self.column_means
        return pd.DataFrame(real_scale_matrix, columns=self.feature_names)