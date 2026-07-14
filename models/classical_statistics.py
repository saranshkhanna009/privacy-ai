import pandas as pd
from copulas.multivariate import GaussianMultivariate
from core.base_model import BaseGenerativeModel
from typing import Dict, Any

class CopulaGenerativeModel(BaseGenerativeModel):
    
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = GaussianMultivariate()
        
    def fit(self, data: pd.DataFrame) -> None:
        print("[INFO] Training Classical Copula Model...")
        training_df = data.copy()
        self.model.fit(training_df)
        self.is_trained = True
        print("[INFO] Classical Copula Model training completed.")

    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Cannot generate data. Model must be fitted first.")
            
        print(f"[INFO] Generating {num_rows} synthetic records via Copula...")
        synthetic_data = self.model.sample(num_rows)
        return synthetic_data