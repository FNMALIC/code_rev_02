from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BaselineModels:
    def __init__(self):
        self.models = {
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, random_state=42))
            ])
        }
        self.trained_models = {}

    def prepare_features(self, feature_dicts: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to numpy array"""
        if not feature_dicts:
            return np.array([])

        # Flatten nested dictionaries
        all_features = []
        feature_names = set()

        for feat_dict in feature_dicts:
            flat_features = self._flatten_dict(feat_dict)
            all_features.append(flat_features)
            feature_names.update(flat_features.keys())

        # Convert to consistent feature vectors
        feature_names = sorted(list(feature_names))
        feature_matrix = []

        for features in all_features:
            feature_vector = [features.get(name, 0) for name in feature_names]
            feature_matrix.append(feature_vector)

        self.feature_names = feature_names
        return np.array(feature_matrix)

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert boolean to int, handle None values
                if isinstance(v, bool):
                    v = int(v)
                elif v is None:
                    v = 0
                elif not isinstance(v, (int, float)):
                    v = 0
                items.append((new_key, v))
        return dict(items)

    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all baseline models"""
        self.trained_models = {}

        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            try:
                model.fit(X, y)
                self.trained_models[name] = model
                logger.info(f"Successfully trained {name}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")

        return self.trained_models

    def predict(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with specified model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else predictions

        return predictions, probabilities

    def save_models(self, path: str):
        """Save all trained models"""
        for name, model in self.trained_models.items():
            model_path = f"{path}/{name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")

    def load_models(self, path: str):
        """Load saved models"""
        for name in self.models.keys():
            model_path = f"{path}/{name}_model.pkl"
            try:
                self.trained_models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load {name} model: {e}")