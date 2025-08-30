from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.class_weight import compute_class_weight
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
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'  # Handle class imbalance
                ))
            ]),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            ),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    probability=True,
                    random_state=42,
                    class_weight='balanced'  # Handle class imbalance
                ))
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

    def _check_class_distribution(self, y: np.ndarray) -> Tuple[bool, Dict]:
        """Check class distribution and handle single class case"""
        unique_classes = np.unique(y)
        class_counts = np.bincount(y.astype(int))

        distribution_info = {
            'unique_classes': unique_classes,
            'class_counts': dict(enumerate(class_counts)),
            'total_samples': len(y)
        }

        # Check if we have at least 2 classes
        has_multiple_classes = len(unique_classes) >= 2

        if not has_multiple_classes:
            logger.warning(f"Only one class found: {unique_classes}. Creating synthetic balanced dataset.")

        return has_multiple_classes, distribution_info

    def _create_balanced_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a more balanced dataset when we have severe class imbalance"""
        unique_classes = np.unique(y)

        if len(unique_classes) == 1:
            # Create synthetic minority class by adding noise to existing samples
            logger.info("Creating synthetic minority class samples...")

            minority_size = max(10, len(X) // 4)  # Create at least 10 samples, up to 25% of majority
            noise_std = np.std(X, axis=0) * 0.1  # 10% of feature std as noise

            # Select random samples to perturb
            indices = np.random.choice(len(X), minority_size, replace=True)
            synthetic_X = X[indices].copy()

            # Add noise to create variation
            for i in range(synthetic_X.shape[1]):
                if noise_std[i] > 0:
                    noise = np.random.normal(0, noise_std[i], minority_size)
                    synthetic_X[:, i] += noise

            # Create opposite labels
            synthetic_y = np.ones(minority_size) if unique_classes[0] == 0 else np.zeros(minority_size)

            # Combine datasets
            X_balanced = np.vstack([X, synthetic_X])
            y_balanced = np.hstack([y, synthetic_y])

            logger.info(
                f"Created {minority_size} synthetic samples. New distribution: {np.bincount(y_balanced.astype(int))}")
            return X_balanced, y_balanced

        return X, y

    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all baseline models with enhanced class handling"""
        self.trained_models = {}

        # Check class distribution
        has_multiple_classes, dist_info = self._check_class_distribution(y)

        logger.info(f"Class distribution: {dist_info['class_counts']}")

        # Handle single class case
        if not has_multiple_classes:
            X, y = self._create_balanced_dataset(X, y)
            has_multiple_classes, dist_info = self._check_class_distribution(y)
            logger.info(f"After balancing - Class distribution: {dist_info['class_counts']}")

        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            try:
                if has_multiple_classes:
                    model.fit(X, y)
                    self.trained_models[name] = model
                    logger.info(f"Successfully trained {name}")
                else:
                    # Fallback: create a dummy classifier that predicts the majority class
                    logger.warning(f"Creating dummy classifier for {name} due to single class")
                    from sklearn.dummy import DummyClassifier
                    dummy = DummyClassifier(strategy='most_frequent')
                    dummy.fit(X, y)
                    self.trained_models[name] = dummy

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                # Create dummy classifier as fallback
                try:
                    from sklearn.dummy import DummyClassifier
                    dummy = DummyClassifier(strategy='most_frequent')
                    dummy.fit(X, y)
                    self.trained_models[name] = dummy
                    logger.info(f"Created dummy classifier for {name} as fallback")
                except Exception as e2:
                    logger.error(f"Failed to create dummy classifier for {name}: {e2}")

        return self.trained_models

    def predict(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with specified model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]
        predictions = model.predict(X)

        # Handle probability predictions
        try:
            probabilities = model.predict_proba(X)
            if probabilities.shape[1] > 1:
                probabilities = probabilities[:, 1]  # Positive class probability
            else:
                probabilities = probabilities[:, 0]  # Single class case
        except (AttributeError, IndexError):
            # Fallback to predictions if no probabilities available
            probabilities = predictions.astype(float)

        return predictions, probabilities

    def save_models(self, path: str):
        """Save all trained models"""
        for name, model in self.trained_models.items():
            model_path = f"{path}/{name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")

        # Save feature names
        if hasattr(self, 'feature_names'):
            features_path = f"{path}/feature_names.pkl"
            joblib.dump(self.feature_names, features_path)
            logger.info(f"Saved feature names to {features_path}")

    def load_models(self, path: str):
        """Load saved models"""
        for name in self.models.keys():
            model_path = f"{path}/{name}_model.pkl"
            try:
                self.trained_models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load {name} model: {e}")

        # Load feature names
        try:
            features_path = f"{path}/feature_names.pkl"
            self.feature_names = joblib.load(features_path)
            logger.info(f"Loaded feature names from {features_path}")
        except Exception as e:
            logger.warning(f"Could not load feature names: {e}")