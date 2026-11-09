"""
Customer Behavioral Segmentation using K-Means and Relative Profiling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional
from src.config import SEGMENT_FEATURES, RANDOM_STATE, SEGMENT_LABELS


class CustomerSegmenter:
    """
    K-Means clustering pipeline with automated relative profile indexing.
    """
    def __init__(self, n_clusters: int = 4, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.feature_names = SEGMENT_FEATURES
        self.is_fitted = False

    def fit_predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Scale features, fit K-Means, and return cluster labels with relative profile matrix.
        """
        X = df[self.feature_names].copy()
        X_scaled = self.scaler.fit_transform(X)
        labels = self.kmeans.fit_predict(X_scaled)
        self.is_fitted = True

        df_clustered = df.copy()
        df_clustered["Cluster"] = labels

        # Absolute profile
        cluster_profile = df_clustered.groupby("Cluster")[self.feature_names].mean()

        # Relative profile indexed against population mean (1.0 = average customer)
        pop_mean = df[self.feature_names].mean()
        relative_profile = cluster_profile.divide(pop_mean).round(2)

        return labels, relative_profile

    def get_cluster_profiles(self, df_with_segments: pd.DataFrame, segment_col: str = "Customer_Segment") -> pd.DataFrame:
        """
        Generate behavioral profile summaries grouped by segment name.
        """
        return df_with_segments.groupby(segment_col)[self.feature_names].mean()
