import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class UtilityAuditor:
    def calculate_correlation_similarity(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        r_clean = real_data.select_dtypes(include=[np.number]).copy()
        s_clean = synthetic_data.select_dtypes(include=[np.number]).copy()
        
        real_corr = r_clean.corr().values
        synth_corr = s_clean.corr().values
        
        return float(np.mean(np.abs(real_corr - synth_corr)))

    def run_tstr_test(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, target_col: str = "Cholesterol") -> float:
        r_clean = real_data.select_dtypes(include=[np.number]).copy()
        s_clean = synthetic_data.select_dtypes(include=[np.number]).copy()
        
        X_synth = s_clean.drop(columns=[target_col])
        y_synth = s_clean[target_col]
        
        X_real = r_clean.drop(columns=[target_col])
        y_real = r_clean[target_col]
        
        _, X_test_real, _, y_test_real = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
        
        evaluator_model = RandomForestRegressor(n_estimators=50, random_state=42)
        evaluator_model.fit(X_synth, y_synth)
        
        predictions = evaluator_model.predict(X_test_real)
        return float(r2_score(y_test_real, predictions))