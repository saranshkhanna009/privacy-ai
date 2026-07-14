import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class PrivacyAuditor:
    
    def __init__(self, threshold_percentile: float = 5.0):
        self.threshold_percentile = threshold_percentile

    def audit_leakage(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict:
        print("[AUDIT] Starting Privacy Proximity Audit...")
        
        r_matrix = real_data.select_dtypes(include=[np.number]).values
        s_matrix = synthetic_data.select_dtypes(include=[np.number]).values

        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(r_matrix)
        distances, _ = nn.kneighbors(s_matrix)
        distances = distances.flatten()

        exact_matches = np.sum(distances < 1e-5)
        min_distance = float(np.min(distances))
        mean_distance = float(np.mean(distances))
        lower_bound = float(np.percentile(distances, self.threshold_percentile))

        passed_audit = exact_matches == 0 and lower_bound > 0.01

        return {
            "passed_audit": passed_audit,
            "exact_duplicates_found": int(exact_matches),
            "min_distance_to_real": min_distance,
            "mean_distance_to_real": mean_distance,
            "5th_percentile_dcr": lower_bound
        }