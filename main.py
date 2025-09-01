import argparse
import ast
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
import re
from src.integration.diff_parser import EnhancedGitDiffParser
from src.utils.ExternalAIReviewer import ExternalAIReviewer
from src.utils.logging_config import setup_logging
from src.data.loader import CodeReviewDataLoader
from src.data.preprocessor import CodePreprocessor
from src.features.ast_features import ASTFeatureExtractor
from src.features.graph_features import GraphFeatureExtractor
from src.models.baselines import BaselineModels
from src.models.codebert_model import CodeBERTMultiTask
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.interpretability import InterpretabilityAnalyzer
from src.integration.git_integration import GitIntegration
from src.features.enhanced_labeling import (
    generate_realistic_labels,
    extract_enhanced_features,
    prepare_improved_baseline_features,
    preprocess_code_fragment,
    analyze_label_distribution
)


import json
import re

ICONS = {
    "assessment": "📝",
    "issues": "⚠️",
    "recommendations": "💡",
    "security": "🔒",
    "performance": "⚡",
    "analysis": "🤖"
}

def pretty_print_external_analysis(ext: dict):
    print("\nExternal AI Insights:")

    for key, value in ext.items():
        if not value or key == "error":
            continue

        title = key.title()
        icon = ICONS.get(key.lower(), "🔹")
        print(f"\n{icon} {title}:")

        # If it's a list, print as bullet points
        if isinstance(value, list):
            for item in value:
                print(f"   • {item}")

        # If it's a dict, pretty print subkeys
        elif isinstance(value, dict):
            for subk, subv in value.items():
                print(f"   {subk.title()}: {subv}")

        # If it's a JSON string with ```json ... ```
        elif isinstance(value, str) and value.strip().startswith("```json"):
            cleaned = re.sub(r"^```json|```$", "", value.strip(), flags=re.MULTILINE).strip()
            try:
                parsed = json.loads(cleaned)
                pretty_print_external_analysis(parsed)  # recursive formatting
            except:
                print(f"   {value}")
        else:
            print(f"   {value}")


logger = logging.getLogger(__name__)


class CompleteCodeReviewSystem:
    """
    Complete automated code review system that integrates all components:
    1. Preprocesses code using CodePreprocessor
    2. Extracts AST and graph features
    3. Uses CodeBERT for deep learning analysis
    4. Provides interpretable feedback
    5. Generates automated review comments
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = Path("models/codebert")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Initialize all components
        self.preprocessor = CodePreprocessor()
        self.ast_extractor = ASTFeatureExtractor()
        self.graph_extractor = GraphFeatureExtractor()
        self.baseline_models = BaselineModels()
        self.codebert_model = CodeBERTMultiTask(
            model_name=config['model']['name'],
            dropout_rate=config['model'].get('dropout', 0.1)
        )
        self.evaluator = ModelEvaluator()
        self.interpretability = InterpretabilityAnalyzer()

        # State tracking
        self.is_trained = False
        self.feature_extractors_ready = False

        # Try to load existing model
        self._load_existing_models()

    def _load_existing_models(self):
        """Load existing trained models if they exist"""
        codebert_path = self.model_path / "codebert_complete.pth"
        baseline_path = "models/baseline"

        # Load CodeBERT model
        if codebert_path.exists():
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.codebert_model.load_state_dict(torch.load(codebert_path, map_location=device))
                self.codebert_model.to(device)
                logger.info("Loaded existing CodeBERT model")
                self.is_trained = True
            except Exception as e:
                logger.warning(f"Failed to load CodeBERT model: {e}")

        # Load baseline models
        if Path(baseline_path).exists():
            try:
                self.baseline_models.load_models(baseline_path)
                logger.info("Loaded existing baseline models")
            except Exception as e:
                logger.warning(f"Failed to load baseline models: {e}")

    def train_all_models(self, commits_data: List[Dict], force_retrain: bool = False) -> Dict[str, Any]:
        """Train all models in the system"""

        # Check if models already exist and we're not forcing retrain
        if self.is_trained and not force_retrain:
            logger.info("Models already trained. Use force_retrain=True to retrain.")
            return {"status": "already_trained", "message": "Models loaded from disk"}

        logger.info("Training complete code review system...")

        # Generate synthetic data if needed
        if len(commits_data) < 50:
            commits_data = self.generate_synthetic_commits(commits_data, target_size=100)

        # Prepare comprehensive features
        X, y, samples = self.prepare_comprehensive_features(commits_data)

        if X.size == 0:
            logger.error("No features extracted. Cannot train models.")
            return {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        train_samples = samples[:len(X_train)]
        test_samples = samples[len(X_train):]

        results = {}

        # 1. Train baseline models (only if not already trained or force retrain)
        if not self.baseline_models.trained_models or force_retrain:
            logger.info("Training baseline models...")
            baseline_models = self.baseline_models.train_all_models(X_train, y_train)
            # Save baseline models
            baseline_path = "models/baseline"
            Path(baseline_path).mkdir(parents=True, exist_ok=True)
            self.baseline_models.save_models(baseline_path)
            results['baseline_models'] = baseline_models
        else:
            logger.info("Using existing baseline models")
            results['baseline_models'] = "loaded_existing"

        # 2. Train CodeBERT model (only if not already trained or force retrain)
        codebert_path = self.model_path / "codebert_complete.pth"
        if not codebert_path.exists() or force_retrain:
            logger.info("Training CodeBERT model...")
            codebert_results = self.train_codebert(train_samples, test_samples)
            results['codebert'] = codebert_results
        else:
            logger.info("Using existing CodeBERT model")
            results['codebert'] = "loaded_existing"

        # 3. Evaluate all models
        logger.info("Evaluating models...")
        evaluation_results = self.evaluate_models(X_test, y_test, test_samples)
        results['evaluation'] = evaluation_results

        # 4. Generate interpretability analysis
        logger.info("Generating interpretability analysis...")
        interpretability_results = self.analyze_interpretability(test_samples[:10])
        results['interpretability'] = interpretability_results

        self.is_trained = True
        return results

    def train_codebert(self, train_samples: List[Dict], test_samples: List[Dict]) -> Dict:
        """Train CodeBERT model with proper setup"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.codebert_model.to(device)
        self.codebert_model.train()

        # Training setup - fix the learning rate type issue
        learning_rate = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])

        optimizer = torch.optim.AdamW(
            self.codebert_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        epochs = self.config['training']['num_epochs']
        batch_size = self.config['training']['batch_size']

        best_loss = float('inf')
        patience = 3
        patience_counter = 0

        # Training loop with early stopping
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(train_samples), batch_size):
                batch = train_samples[i:i + batch_size]

                try:
                    # Prepare batch
                    input_ids = torch.cat([s['tokenized']['input_ids'] for s in batch]).to(device)
                    attention_mask = torch.cat([s['tokenized']['attention_mask'] for s in batch]).to(device)
                    labels = torch.tensor([s['is_high_quality'] for s in batch], dtype=torch.float).to(device)

                    # Forward pass
                    outputs = self.codebert_model(input_ids, attention_mask, task='quality')
                    loss = torch.nn.BCELoss()(outputs['quality_score'].squeeze(), labels)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.warning(f"Error in training batch: {e}")
                    continue

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

                # Early stopping logic
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.codebert_model.state_dict(), self.model_path / "codebert_complete.pth")
                    logger.info("Saved new best model")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.codebert_model.load_state_dict(torch.load(self.model_path / "codebert_complete.pth"))
        logger.info("Loaded best model weights")

        return {'trained': True, 'model_path': str(self.model_path), 'best_loss': best_loss}
    def prepare_comprehensive_features(self, commits_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Combine all feature extraction methods for comprehensive analysis"""
        logger.info("Extracting comprehensive features using all extractors...")

        all_samples = []

        for commit in tqdm(commits_data, desc="Processing commits"):
            commit_context = {
                'files': commit.get('files_changed', []),
                'hunks': commit.get('hunks', [])
            }

            for hunk in commit['hunks']:
                if not any(line.strip() for line in hunk['added_lines']):
                    continue

                try:
                    code = '\n'.join(hunk['added_lines'])

                    # Skip empty or whitespace-only code
                    if not code.strip():
                        continue

                    # 1. Preprocess code
                    processed_code = self.preprocessor.preprocess_code(code)

                    # Skip if preprocessing failed
                    if not processed_code or not processed_code.strip():
                        continue

                    tokenized = self.preprocessor.tokenize_code(processed_code)

                    # 2. Extract AST features
                    ast_features = self.ast_extractor.extract_features(code)

                    # 3. Extract graph features
                    graph_features = self.graph_extractor.extract_features(code)

                    # 4. Extract enhanced features (from previous work)
                    enhanced_features = extract_enhanced_features(hunk, commit_context)

                    # 5. Generate realistic quality labels
                    quality_score, is_high_quality, issues = generate_realistic_labels(hunk)

                    # Combine all features
                    combined_features = {
                        **ast_features,
                        **graph_features,
                        **enhanced_features,
                        'tokenized_length': len(tokenized['input_ids'][0]),
                        'has_syntax_errors': 0 if processed_code else 1
                    }

                    sample = {
                        'features': combined_features,
                        'code': code,
                        'processed_code': processed_code,
                        'tokenized': tokenized,
                        'quality_score': quality_score,
                        'is_high_quality': is_high_quality,
                        'issues': issues,
                        'hunk_data': hunk,
                        'commit_data': commit
                    }

                    all_samples.append(sample)

                except Exception as e:
                    logger.warning(f"Error processing hunk: {e}")
                    continue

        if not all_samples:
            return np.array([]), np.array([]), []

        # Extract features and labels
        X = self.baseline_models.prepare_features([s['features'] for s in all_samples])
        y = np.array([s['is_high_quality'] for s in all_samples])

        logger.info(f"Extracted {len(all_samples)} samples with {X.shape[1]} features")
        logger.info(f"Quality distribution: {np.mean(y):.1%} high quality")

        return X, y, all_samples

    # def train_all_models(self, commits_data: List[Dict]) -> Dict[str, Any]:
    #     """Train all models in the system"""
    #     logger.info("Training complete code review system...")
    #
    #     # Generate synthetic data if needed
    #     if len(commits_data) < 50:
    #         commits_data = self.generate_synthetic_commits(commits_data, target_size=100)
    #
    #     # Prepare comprehensive features
    #     X, y, samples = self.prepare_comprehensive_features(commits_data)
    #
    #     if X.size == 0:
    #         logger.error("No features extracted. Cannot train models.")
    #         return {}
    #
    #     # Split data
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.2, random_state=42,
    #         stratify=y if len(np.unique(y)) > 1 else None
    #     )
    #
    #     train_samples = samples[:len(X_train)]
    #     test_samples = samples[len(X_train):]
    #
    #     results = {}
    #
    #     # 1. Train baseline models
    #     logger.info("Training baseline models...")
    #     baseline_models = self.baseline_models.train_all_models(X_train, y_train)
    #     results['baseline_models'] = baseline_models
    #
    #     # 2. Train CodeBERT model
    #     logger.info("Training CodeBERT model...")
    #     codebert_results = self.train_codebert(train_samples, test_samples)
    #     results['codebert'] = codebert_results
    #
    #     # 3. Evaluate all models
    #     logger.info("Evaluating models...")
    #     evaluation_results = self.evaluate_models(X_test, y_test, test_samples)
    #     results['evaluation'] = evaluation_results
    #
    #     # 4. Generate interpretability analysis
    #     logger.info("Generating interpretability analysis...")
    #     interpretability_results = self.analyze_interpretability(test_samples[:10])
    #     results['interpretability'] = interpretability_results
    #
    #     self.is_trained = True
    #     return results
    #
    # def train_codebert(self, train_samples: List[Dict], test_samples: List[Dict]) -> Dict:
    #     """Train CodeBERT model with proper setup"""
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     self.codebert_model.to(device)
    #     self.codebert_model.train()
    #
    #     # Training setup - fix the learning rate type issue
    #     learning_rate = float(self.config['training']['learning_rate'])
    #     weight_decay = float(self.config['training']['weight_decay'])
    #
    #     optimizer = torch.optim.AdamW(
    #         self.codebert_model.parameters(),
    #         lr=learning_rate,
    #         weight_decay=weight_decay
    #     )
    #
    #     epochs = self.config['training']['num_epochs']
    #     batch_size = self.config['training']['batch_size']
    #
    #     # Training loop
    #     for epoch in range(epochs):
    #         total_loss = 0
    #         num_batches = 0
    #
    #         for i in range(0, len(train_samples), batch_size):
    #             batch = train_samples[i:i + batch_size]
    #
    #             try:
    #                 # Prepare batch
    #                 input_ids = torch.cat([s['tokenized']['input_ids'] for s in batch]).to(device)
    #                 attention_mask = torch.cat([s['tokenized']['attention_mask'] for s in batch]).to(device)
    #                 labels = torch.tensor([s['is_high_quality'] for s in batch], dtype=torch.float).to(device)
    #
    #                 # Forward pass
    #                 outputs = self.codebert_model(input_ids, attention_mask, task='quality')
    #                 loss = torch.nn.BCELoss()(outputs['quality_score'].squeeze(), labels)
    #
    #                 # Backward pass
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
    #
    #                 total_loss += loss.item()
    #                 num_batches += 1
    #
    #             except Exception as e:
    #                 logger.warning(f"Error in training batch: {e}")
    #                 continue
    #
    #         if num_batches > 0:
    #             avg_loss = total_loss / num_batches
    #             logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    #
    #     # Save model
    #     model_path = Path("models/codebert")
    #     model_path.mkdir(parents=True, exist_ok=True)
    #     torch.save(self.codebert_model.state_dict(), model_path / "codebert_complete.pth")
    #
    #     return {'trained': True, 'model_path': str(model_path)}

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, test_samples: List[Dict]) -> Dict:
        """Evaluate all models comprehensively"""
        results = {}

        # Evaluate baseline models
        for model_name, model in self.baseline_models.trained_models.items():
            try:
                y_pred, y_prob = self.baseline_models.predict(model_name, X_test)
                metrics = self.evaluator.evaluate_classification(y_test, y_pred, y_prob)
                results[f'baseline_{model_name}'] = metrics
                logger.info(f"{model_name}: F1={metrics['f1']:.3f}, AUC={metrics.get('auc', 0):.3f}")
            except Exception as e:
                logger.warning(f"Error evaluating {model_name}: {e}")

        # Evaluate CodeBERT
        try:
            codebert_predictions = self.evaluate_codebert(test_samples)
            results['codebert'] = codebert_predictions
        except Exception as e:
            logger.warning(f"Error evaluating CodeBERT: {e}")

        return results

    def evaluate_codebert(self, test_samples: List[Dict]) -> Dict:
        """Evaluate CodeBERT model"""
        self.codebert_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        predictions = []
        true_labels = []

        with torch.no_grad():
            for sample in test_samples[:20]:  # Limit for memory
                try:
                    input_ids = sample['tokenized']['input_ids'].to(device)
                    attention_mask = sample['tokenized']['attention_mask'].to(device)

                    outputs = self.codebert_model(input_ids, attention_mask, task='quality')
                    pred = outputs['quality_score'].cpu().numpy().flatten()[0]

                    predictions.append(pred)
                    true_labels.append(sample['is_high_quality'])

                except Exception as e:
                    logger.warning(f"Error evaluating sample: {e}")
                    continue

        if predictions:
            y_pred_binary = [1 if p > 0.5 else 0 for p in predictions]
            metrics = self.evaluator.evaluate_classification(
                np.array(true_labels),
                np.array(y_pred_binary),
                np.array(predictions)
            )
            return metrics

        return {'error': 'No predictions generated'}

    def analyze_interpretability(self, samples: List[Dict]) -> Dict:
        """Generate interpretability analysis for sample predictions"""
        interpretability_results = {}

        for i, sample in enumerate(samples):
            try:
                # Get model predictions
                X_sample = self.baseline_models.prepare_features([sample['features']])

                # Analyze feature importance for this sample
                if 'random_forest' in self.baseline_models.trained_models:
                    rf_model = self.baseline_models.trained_models['random_forest']
                    feature_importance = self.interpretability.generate_feature_importance(rf_model, X_sample, list(
                        sample['features'].keys()))

                    # Get top features
                    if feature_importance:
                        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    else:
                        top_features = []

                    interpretability_results[f'sample_{i}'] = {
                        'code_snippet': sample['code'][:200] + '...' if len(sample['code']) > 200 else sample['code'],
                        'predicted_quality': sample['is_high_quality'],
                        'quality_score': sample['quality_score'],
                        'key_issues': sample['issues'],
                        'top_features': top_features
                    }

            except Exception as e:
                logger.warning(f"Error analyzing sample {i}: {e}")
                continue

        return interpretability_results

    def generate_review_comments(self, code: str, hunk_data: Dict = None) -> List[str]:
        """
        Generate automated review comments for given code
        This is the core automated review functionality
        """
        if not self.is_trained:
            return ["Error: System not trained yet"]

        try:
            # 1. Preprocess and analyze code
            processed_code = self.preprocessor.preprocess_code(code)
            tokenized = self.preprocessor.tokenize_code(processed_code)

            # 2. Extract features
            ast_features = self.ast_extractor.extract_features(code)
            graph_features = self.graph_extractor.extract_features(code)

            if hunk_data:
                enhanced_features = extract_enhanced_features(hunk_data, {})
            else:
                enhanced_features = {}

            # Combine features
            combined_features = {**ast_features, **graph_features, **enhanced_features}
            X = self.baseline_models.prepare_features([combined_features])

            # 3. Generate predictions from multiple models
            comments = []

            # Baseline model predictions
            for model_name, model in self.baseline_models.trained_models.items():
                try:
                    y_pred, y_prob = self.baseline_models.predict(model_name, X)
                    confidence = y_prob[0] if len(y_prob) > 0 else 0

                    if y_pred[0] == 0 and confidence > 0.7:  # Low quality with high confidence
                        comments.append(
                            f"⚠️ {model_name.title()} model flags this code as potentially problematic (confidence: {confidence:.2f})")
                except Exception as e:
                    continue

            # CodeBERT prediction
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.codebert_model.eval()

                with torch.no_grad():
                    input_ids = tokenized['input_ids'].to(device)
                    attention_mask = tokenized['attention_mask'].to(device)

                    outputs = self.codebert_model(input_ids, attention_mask, task='quality')
                    codebert_score = outputs['quality_score'].cpu().numpy().flatten()[0]

                    if codebert_score < 0.6:
                        comments.append(
                            f"🤖 CodeBERT analysis suggests code quality concerns (score: {codebert_score:.2f})")
                    elif codebert_score > 0.8:
                        comments.append(
                            f"✅ CodeBERT analysis indicates good code quality (score: {codebert_score:.2f})")

            except Exception as e:
                logger.warning(f"CodeBERT prediction failed: {e}")

            # 4. Generate specific issue-based comments
            if hunk_data:
                _, _, issues = generate_realistic_labels(hunk_data)

                issue_messages = {
                    'very_long_change': "📏 Consider breaking this large change into smaller, more focused commits",
                    'no_comments': "📝 Adding comments would improve code readability and maintainability",
                    'debug_prints': "🐛 Remove debug print statements before committing",
                    'long_lines': "📐 Some lines exceed recommended length. Consider breaking them up",
                    'deep_nesting': "🔄 Deep nesting detected. Consider refactoring to reduce complexity",
                    'unparseable_syntax': "❌ Syntax errors detected. Please review the code",
                    'todo_markers': "📋 TODO/FIXME comments found. Consider addressing them"
                }

                for issue in issues:
                    if issue in issue_messages:
                        comments.append(issue_messages[issue])

            # 5. Generate suggestions using AST analysis
            if ast_features.get('cyclomatic_complexity', 0) > 10:
                comments.append("🔧 High cyclomatic complexity detected. Consider refactoring into smaller functions")

            if ast_features.get('num_functions', 0) > 5:
                comments.append("🏗️ Multiple functions in one change. Consider separating concerns")

            # 6. Default positive feedback if no issues
            if not comments:
                comments.append("✅ Code looks good! No significant issues detected")

            return comments

        except Exception as e:
            logger.error(f"Error generating review comments: {e}")
            return ["Error generating automated review"]

    def generate_synthetic_commits(self, commits_data: List[Dict], target_size: int = 100) -> List[Dict]:
        """Generate synthetic commits for training"""
        logger.info(f"Generating synthetic commits to reach {target_size} total")

        synthetic_templates = [
            # High quality examples
            ("high", """def process_user_data(user_input: dict) -> dict:
    \"\"\"
    Process and validate user input data.

    Args:
        user_input: Dictionary containing user data

    Returns:
        Processed user data dictionary

    Raises:
        ValueError: If input data is invalid
    \"\"\"
    if not isinstance(user_input, dict):
        raise ValueError("Input must be a dictionary")

    processed_data = {}

    # Validate required fields
    required_fields = ['name', 'email']
    for field in required_fields:
        if field not in user_input:
            raise ValueError(f"Missing required field: {field}")
        processed_data[field] = str(user_input[field]).strip()

    # Optional fields with defaults
    processed_data['age'] = int(user_input.get('age', 0))
    processed_data['active'] = bool(user_input.get('active', True))

    return processed_data"""),

            # Low quality examples
            ("low", """def func(x):
    print(x)
    if x>10:
        print("big")
        return x*2
    else:
        print("small")
        return x*3"""),

            ("low", """global_var = []

def add_stuff(thing):
    global global_var
    global_var.append(thing)
    print(global_var)
    return len(global_var)"""),
        ]

        synthetic_commits = list(commits_data)

        while len(synthetic_commits) < target_size:
            for quality, code in synthetic_templates:
                if len(synthetic_commits) >= target_size:
                    break

                # Add variations
                varied_code = code.replace('user', f'user_{len(synthetic_commits)}')

                synthetic_hunk = {
                    'added_lines': varied_code.split('\n'),
                    'removed_lines': [],
                    'context_lines': ['# Context', ''],
                    'line_start': 1,
                    'line_end': len(varied_code.split('\n'))
                }

                synthetic_commit = {
                    'id': f'synthetic_{len(synthetic_commits)}',
                    'message': f'Synthetic {quality} quality commit',
                    'author': 'synthetic_generator',
                    'files_changed': ['synthetic_file.py'],
                    'hunks': [synthetic_hunk]
                }

                synthetic_commits.append(synthetic_commit)

        return synthetic_commits


class ImprovedCodeReviewSystem(CompleteCodeReviewSystem):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Initialize external AI reviewer
        api_key = config.get('external_ai', {}).get('api_key')
        # model = config.get('external_ai', {}).get('model', 'gpt-3.5-turbo')
        self.external_ai = ExternalAIReviewer(api_key)

        # Enhanced diff parser
        self.diff_parser = EnhancedGitDiffParser()

    def generate_comprehensive_review(self, hunk: Dict, file_path: str, show_all_lines: bool = True) -> Dict:
        """Generate comprehensive review using multiple analysis methods"""

        added_code = '\n'.join(hunk['added_lines']) if hunk['added_lines'] else ""
        removed_code = '\n'.join(hunk['removed_lines']) if hunk['removed_lines'] else ""
        context = '\n'.join(hunk['context_lines']) if hunk['context_lines'] else ""

        review_results = {
            'local_analysis': {},
            'external_analysis': {},
            'combined_feedback': []
        }

        # 1. Local model analysis (existing CodeBERT + baselines)
        if added_code.strip():
            local_comments = self.generate_review_comments(added_code, hunk)
            review_results['local_analysis'] = {
                'comments': local_comments,
                'code_lines_analyzed': len(hunk['added_lines']),
                'total_lines_available': len(hunk['added_lines'])
            }

        # 2. External AI analysis
        if self.external_ai.api_key and (added_code.strip() or removed_code.strip()):
            external_analysis = self.external_ai.analyze_code_change(
                added_code, removed_code, context, file_path
            )
            review_results['external_analysis'] = external_analysis

        # 3. Combined analysis
        combined_feedback = self._combine_analysis_results(
            review_results['local_analysis'],
            review_results['external_analysis'],
            added_code,
            removed_code,
            hunk
        )
        review_results['combined_feedback'] = combined_feedback

        return review_results

    def _combine_analysis_results(self, local: Dict, external: Dict, added: str, removed: str, hunk: Dict) -> List[str]:
        """Combine local and external analysis into actionable feedback"""
        feedback = []

        # Change type analysis
        if removed and added:
            feedback.append(f"Modified {len(removed.split())} -> {len(added.split())} lines")
        elif added and not removed:
            feedback.append(f"Added {len(added.splitlines())} new lines")
        elif removed and not added:
            feedback.append(f"Removed {len(removed.splitlines())} lines")

        # Local analysis feedback
        if local.get('comments'):
            feedback.extend(local['comments'])

        # External AI feedback
        if external and 'error' not in external:
            if external.get('assessment'):
                feedback.append(f"AI Assessment: {external['assessment']}")

            if external.get('issues'):
                feedback.append(f"Potential Issues: {external['issues']}")

            if external.get('recommendations'):
                feedback.append(f"Recommendations: {external['recommendations']}")

            if external.get('security'):
                feedback.append(f"Security: {external['security']}")

            if external.get('performance'):
                feedback.append(f"Performance: {external['performance']}")

        # Complexity analysis
        if added:
            lines = added.splitlines()
            if len(lines) > 20:
                feedback.append("Large change detected - consider breaking into smaller commits")

            # Check for specific patterns
            if 'TODO' in added or 'FIXME' in added:
                feedback.append("Contains TODO/FIXME - ensure these are addressed")

            if 'print(' in added or 'console.log' in added:
                feedback.append("Debug statements detected - remove before production")

        return feedback if feedback else ["No specific issues detected"]

# def run_automated_review(system: CompleteCodeReviewSystem, repo_path: str, target_files: List[str] = None):
#     """
#     Run automated review on specific files or recent changes
#     This demonstrates the main automated review workflow
#     """
#     logger.info("Running automated code review...")
#
#     git_integration = GitIntegration(repo_path)
#
#     # Get recent changes or specified files
#     if target_files:
#         # Review specific files
#         for file_path in target_files:
#             logger.info(f"Reviewing file: {file_path}")
#             try:
#                 with open(Path(repo_path) / file_path, 'r') as f:
#                     code = f.read()
#
#                 comments = system.generate_review_comments(code)
#
#                 print(f"\n{'=' * 50}")
#                 print(f"AUTOMATED REVIEW: {file_path}")
#                 print('=' * 50)
#                 for i, comment in enumerate(comments, 1):
#                     print(f"{i}. {comment}")
#                 print()
#
#             except Exception as e:
#                 logger.error(f"Error reviewing {file_path}: {e}")
#     else:
#         # Review recent changes (staged or recent commits)
#         diff = git_integration.get_staged_diff()
#         if diff:
#             logger.info("Reviewing staged changes...")
#
#             # Parse diff into structured data
#             parsed_files = git_integration.diff_parser.parse_diff(diff)
#
#             if not parsed_files:
#                 print("\n📝 No parseable changes found in staged diff.")
#                 return
#
#             print(f"\n🔍 AUTOMATED REVIEW OF STAGED CHANGES")
#             print(f"Found {len(parsed_files)} files with changes")
#             print("=" * 60)
#
#             for file_data in parsed_files:
#                 file_path = file_data['file_path']
#                 print(f"\n📁 FILE: {file_path}")
#                 print("-" * 50)
#
#                 for i, hunk in enumerate(file_data['hunks']):
#                     if not hunk['added_lines']:
#                         continue
#
#                     print(f"\n🔄 Hunk {i + 1} (lines {hunk['line_start']}-{hunk['line_end']}):")
#
#                     # Show the added code
#                     added_code = '\n'.join(hunk['added_lines'])
#                     if added_code.strip():
#                         print("Added code:")
#                         for j, line in enumerate(hunk['added_lines'][:5]):  # Show first 5 lines
#                             print(f"  + {line}")
#
#                         if len(hunk['added_lines']) > 5:
#                             print(f"    ... and {len(hunk['added_lines']) - 5} more lines")
#
#                     # Generate review comments for this hunk
#                     try:
#                         comments = system.generate_review_comments(added_code, hunk)
#
#                         print("\n💬 Review Comments:")
#                         for j, comment in enumerate(comments, 1):
#                             print(f"  {j}. {comment}")
#
#                     except Exception as e:
#                         logger.error(f"Error reviewing hunk: {e}")
#                         print(f"  ❌ Error analyzing this hunk: {e}")
#
#                     print()
#         else:
#             print("\n📝 No staged changes found. Please stage changes or specify files to review.")


def run_automated_review_enhanced(system: ImprovedCodeReviewSystem, repo_path: str, target_files: List[str] = None):
    """Enhanced automated review with comprehensive analysis"""
    logger.info("Running enhanced automated code review...")

    git_integration = GitIntegration(repo_path)
    git_integration.diff_parser = system.diff_parser  # Use enhanced parser

    if target_files:
        # Review specific files (existing logic)
        for file_path in target_files:
            logger.info(f"Reviewing file: {file_path}")
            try:
                with open(Path(repo_path) / file_path, 'r') as f:
                    code = f.read()

                print(f"\n{'=' * 60}")
                print(f"COMPREHENSIVE REVIEW: {file_path}")
                print('=' * 60)

                # Create mock hunk for full file analysis
                mock_hunk = {
                    'added_lines': code.splitlines(),
                    'removed_lines': [],
                    'context_lines': [],
                    'line_start': 1,
                    'line_end': len(code.splitlines())
                }

                review = system.generate_comprehensive_review(mock_hunk, file_path)

                # Display results
                for i, comment in enumerate(review['combined_feedback'], 1):
                    print(f"{i}. {comment}")

            except Exception as e:
                logger.error(f"Error reviewing {file_path}: {e}")

    else:
        # Review staged changes
        diff = git_integration.get_staged_diff()
        if diff:
            parsed_files = git_integration.diff_parser.parse_diff(diff)

            if not parsed_files:
                print("\nNo parseable changes found.")
                return

            print(f"\nCOMPREHENSIVE REVIEW OF STAGED CHANGES")
            print(f"Analyzing {len(parsed_files)} files")
            print("=" * 70)

            for file_data in parsed_files:
                file_path = file_data['file_path']
                print(f"\nFILE: {file_path}")
                print("-" * 50)

                if not file_data['hunks']:
                    print("No processable hunks found in this file")
                    continue

                for i, hunk in enumerate(file_data['hunks']):
                    print(f"\nHunk {i + 1} (lines {hunk.get('line_start', '?')}-{hunk.get('line_end', '?')}):")

                    # Show ALL changes, not just first 5
                    if hunk['added_lines']:
                        print("Added:")
                        for line in hunk['added_lines']:
                            print(f"  + {line}")

                    if hunk['removed_lines']:
                        print("Removed:")
                        for line in hunk['removed_lines']:
                            print(f"  - {line}")

                    # Generate comprehensive review
                    review = system.generate_comprehensive_review(hunk, file_path)

                    print("\nComprehensive Analysis:")
                    for j, feedback in enumerate(review['combined_feedback'], 1):
                        print(f"  {j}. {feedback}")

                    # Show external AI analysis separately if available
                    if review['external_analysis'] and 'error' not in review['external_analysis']:
                        pretty_print_external_analysis(review['external_analysis'])
                    print()
        else:
            print("\nNo staged changes found.")
def main():
    parser = argparse.ArgumentParser(description="Complete Automated Code Review System")
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--repo-path', required=True, help='Path to git repository')
    parser.add_argument('--task', choices=['train', 'review', 'both'], default='both')
    parser.add_argument('--files', nargs='*', help='Specific files to review')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--force-retrain',default=False, action='store_true', help='Force retraining even if models exist')
    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    config = load_config(args.config)

    # Initialize system
    # system = CompleteCodeReviewSystem(config)

    system = ImprovedCodeReviewSystem(config)

    if args.task in ['train', 'both']:
        # Load training data
        data_loader = CodeReviewDataLoader(args.repo_path)
        commits_data = data_loader.extract_commits(limit=config.get('data_limit', 500))
        force_retrain = args.force_retrain
        if not commits_data:
            logger.error("No commit data found for training")
            return

        # Train the system

        training_results = system.train_all_models(commits_data, force_retrain=force_retrain)

        # Save results
        results_path = Path("results")
        results_path.mkdir(exist_ok=True)

        with open(results_path / "complete_training_results.json", 'w') as f:
            json.dump(training_results, f, indent=2, default=str)

        logger.info("Training completed successfully")

        # Create visualizations
        if args.visualize:
            logger.info("Creating visualizations...")
            # Visualization code would go here

    if args.task in ['review', 'both']:
        if not system.is_trained and args.task == 'review':
            logger.error("System not trained. Please train first or use --task both")
            return

        # Run automated review
        run_automated_review_enhanced(system, args.repo_path, args.files)

    logger.info("Complete code review system finished")


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    main()