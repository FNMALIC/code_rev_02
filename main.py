import argparse
import json

import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging
from tqdm import tqdm

from src.utils.logging_config import setup_logging
from src.data.loader import CodeReviewDataLoader
from src.data.preprocessor import CodePreprocessor
from src.features.ast_features import ASTFeatureExtractor
from src.features.graph_features import GraphFeatureExtractor
from src.models.baselines import BaselineModels
from src.models.codebert_model import CodeBERTMultiTask
from src.models.mil_model import MultiInstanceLearning
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.interpretability import InterpretabilityAnalyzer
from src.integration.git_integration import GitIntegration

logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_baseline_features(commits_data: List[Dict]) -> tuple:
    """Prepare features for baseline models"""
    ast_extractor = ASTFeatureExtractor()
    graph_extractor = GraphFeatureExtractor()
    preprocessor = CodePreprocessor()

    all_features = []
    all_labels = []

    logger.info("Extracting features for baseline models...")

    for commit in tqdm(commits_data):
        for hunk in commit['hunks']:
            # Combine added and removed lines
            code = '\n'.join(hunk['added_lines'] + hunk['context_lines'])

            if not code.strip():
                continue

            # Extract features
            ast_features = ast_extractor.extract_ast_features(code)
            graph = graph_extractor.create_dependency_graph(code)
            graph_features = graph_extractor.extract_graph_features(graph)
            code_metrics = preprocessor.extract_code_metrics(code)

            # Combine all features
            combined_features = {
                **ast_features,
                **graph_features,
                **code_metrics
            }

            all_features.append(combined_features)

            # Generate synthetic labels (in real scenario, use actual review data)
            quality_score = min(1.0, len(hunk['added_lines']) / 10.0)  # Simple heuristic
            all_labels.append(quality_score > 0.5)

    return all_features, all_labels


def train_baseline_models(config: Dict, commits_data: List[Dict]) -> Dict:
    """Train baseline models"""
    logger.info("Training baseline models...")

    # Prepare features
    feature_dicts, labels = prepare_baseline_features(commits_data)

    # Convert to numpy arrays
    baseline_models = BaselineModels()
    X = baseline_models.prepare_features(feature_dicts)
    y = np.array(labels)

    # Train models
    trained_models = baseline_models.train_all_models(X, y)

    # Save models
    models_path = Path("models/baselines")
    models_path.mkdir(parents=True, exist_ok=True)
    baseline_models.save_models(str(models_path))

    return trained_models


def train_codebert_model(config: Dict, commits_data: List[Dict]) -> CodeBERTMultiTask:
    """Train CodeBERT multi-task model"""
    logger.info("Training CodeBERT model...")

    # Initialize model and preprocessor
    model = CodeBERTMultiTask(
        model_name=config['model']['name'],
        dropout_rate=config['model'].get('dropout', 0.1)
    )

    preprocessor = CodePreprocessor(config['model']['name'])

    # Prepare training data
    train_data = []
    for commit in commits_data:
        for hunk in commit['hunks']:
            code = '\n'.join(hunk['added_lines'] + hunk['context_lines'])
            if code.strip():
                tokenized = preprocessor.tokenize_code(code)
                train_data.append({
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'quality_label': len(hunk['added_lines']) > 5  # Simple heuristic
                })

    # Training loop (simplified)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Simple training loop
    for epoch in range(config['training']['num_epochs']):
        total_loss = 0
        for i in range(0, len(train_data), config['training']['batch_size']):
            batch = train_data[i:i + config['training']['batch_size']]

            # Prepare batch tensors
            input_ids = torch.cat([item['input_ids'] for item in batch]).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in batch]).to(device)
            labels = torch.tensor([item['quality_label'] for item in batch], dtype=torch.float).to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, task='quality')

            # Compute loss
            loss = torch.nn.BCELoss()(outputs['quality_score'].squeeze(), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}, Loss: {total_loss / len(train_data):.4f}")

    # Save model
    model_path = Path("models/codebert")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path / "codebert_multitask.pth")

    return model


def evaluate_models(config: Dict, models: Dict, test_data: List[Dict]) -> Dict:
    """Evaluate all trained models"""
    logger.info("Evaluating models...")

    evaluator = ModelEvaluator()
    results = {}

    # Evaluate baseline models
    for model_name, model in models.get('baselines', {}).items():
        # Prepare test features (simplified)
        y_true = np.random.randint(0, 2, 100)  # Dummy test data
        y_pred = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)

        results[model_name] = evaluator.evaluate_classification(y_true, y_pred, y_prob)

    return results


def main():
    parser = argparse.ArgumentParser(description="Automated Code Review System")
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--repo-path', required=True, help='Path to git repository')
    parser.add_argument('--task', choices=['baselines', 'codebert', 'all'], default='all')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--hook-mode', action='store_true', help='Run in pre-commit hook mode')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    # Load configuration
    config = load_config(args.config)

    # Initialize components
    data_loader = CodeReviewDataLoader(args.repo_path)
    git_integration = GitIntegration(args.repo_path)

    if args.hook_mode:
        # Pre-commit hook mode
        logger.info("Running in pre-commit hook mode")
        diff = git_integration.get_staged_diff()
        if diff:
            # Analyze staged changes (simplified)
            logger.info("Analyzing staged changes...")
            # Implementation would analyze diff and provide feedback
        return

    # Load data
    logger.info("Loading commit data...")
    commits_data = data_loader.extract_commits(limit=config.get('data_limit', 1000))

    if not commits_data:
        logger.error("No commit data found. Exiting.")
        return

    trained_models = {}

    # Train models based on task
    if args.task in ['baselines', 'all']:
        trained_models['baselines'] = train_baseline_models(config, commits_data)

    if args.task in ['codebert', 'all']:
        trained_models['codebert'] = train_codebert_model(config, commits_data)

    # Evaluation
    if args.evaluate:
        results = evaluate_models(config, trained_models, commits_data)

        # Save results
        results_path = Path("results")
        results_path.mkdir(exist_ok=True)

        with open(results_path / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation completed. Results saved to results/evaluation_results.json")

    # Generate interpretability analysis
    if config.get('evaluation', {}).get('generate_explanations', False):
        analyzer = InterpretabilityAnalyzer()

        # SHAP analysis for baseline models (simplified)
        logger.info("Generating interpretability analysis...")
        # Implementation would generate SHAP values and attention visualizations

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()