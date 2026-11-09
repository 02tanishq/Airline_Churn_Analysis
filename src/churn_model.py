"""
Churn Prediction Modeling: Training, Cross-Validation, Evaluation & Explainability.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.config import MODEL_FEATURES, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, XGB_PARAMS, MODELS_DIR


class ChurnPredictor:
    """
    Supervised Churn Model Trainer & Evaluator.
    """
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.features = MODEL_FEATURES
        self.target = TARGET_COLUMN
        self.scaler = StandardScaler()
        self.best_model = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into training and test sets.
        """
        X = df[self.features]
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test

    def benchmark_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
        """
        Train and evaluate Logistic Regression, Random Forest, and XGBoost.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=self.random_state),
            "XGBoost": XGBClassifier(**XGB_PARAMS)
        }

        results = []
        for name, model in models.items():
            if name == "Logistic Regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob)
            }
            results.append(metrics)

            if name == "XGBoost":
                self.best_model = model

        return pd.DataFrame(results)

    def extract_feature_importance(self) -> pd.DataFrame:
        """
        Extract ranked feature importances from the trained XGBoost model.
        """
        if self.best_model is None:
            raise ValueError("Model is not trained yet. Call benchmark_models first.")

        importance_df = pd.DataFrame({
            "Feature": self.features,
            "Importance": self.best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        return importance_df

    def save_model(self, file_path: Path = MODELS_DIR / "xgb_model.pkl") -> None:
        """
        Serialize model weights.
        """
        if self.best_model is not None:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.best_model, file_path)

    @staticmethod
    def load_model(file_path: Path = MODELS_DIR / "xgb_model.pkl") -> Any:
        """
        Load serialized model.
        """
        return joblib.load(file_path)
