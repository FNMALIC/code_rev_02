import shap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class InterpretabilityAnalyzer:
    def __init__(self, save_path: str = "results/interpretability"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def generate_shap_explanations(self,
                                   model: Any,
                                   X_test: np.ndarray,
                                   feature_names: List[str],
                                   sample_size: int = 100) -> shap.Explanation:
        """Generate SHAP explanations for model predictions"""
        try:
            # Sample data if too large
            if len(X_test) > sample_size:
                indices = np.random.choice(len(X_test), sample_size, replace=False)
                X_sample = X_test[indices]
            else:
                X_sample = X_test

            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_sample)

            shap_values = explainer(X_sample)

            # Save summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(self.save_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Save feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(self.save_path / 'shap_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("SHAP explanations generated successfully")
            return shap_values

        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return None

    def visualize_attention_weights(self,
                                    attention_weights: torch.Tensor,
                                    tokens: List[str],
                                    save_name: str = "attention_weights") -> None:
        """Create attention heatmaps"""
        try:
            # Convert to numpy if tensor
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()

            # Handle multi-head attention - average across heads
            if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
                attention_weights = attention_weights.mean(axis=1)  # Average across heads

            # Take first sample if batch
            if len(attention_weights.shape) == 3:
                attention_weights = attention_weights[0]

            # Truncate tokens if too long
            max_len = min(len(tokens), attention_weights.shape[0])
            tokens = tokens[:max_len]
            attention_weights = attention_weights[:max_len, :max_len]

            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                attention_weights,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                annot=False,
                fmt='.2f'
            )
            plt.title('Attention Weights Visualization')
            plt.xlabel('Target Tokens')
            plt.ylabel('Source Tokens')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.save_path / f'{save_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Attention visualization saved as {save_name}.png")

        except Exception as e:
            logger.error(f"Error creating attention visualization: {e}")

    def generate_feature_importance(self,
                                    model: Any,
                                    X_test: np.ndarray,
                                    feature_names: List[str]) -> Dict[str, float]:
        """Generate feature importance scores"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))

                # Plot feature importance
                plt.figure(figsize=(10, 8))
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:20]  # Top 20 features

                features, scores = zip(*top_features)
                y_pos = np.arange(len(features))

                plt.barh(y_pos, scores)
                plt.yticks(y_pos, features)
                plt.xlabel('Feature Importance')
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig(self.save_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()

                return importance_dict

            else:
                # Use SHAP for other models
                logger.info("Using SHAP for feature importance")
                shap_values = self.generate_shap_explanations(model, X_test, feature_names)
                if shap_values is not None:
                    importances = np.abs(shap_values.values).mean(0)
                    return dict(zip(feature_names, importances))

        except Exception as e:
            logger.error(f"Error generating feature importance: {e}")

        return {}

    def analyze_mil_attention(self,
                              attention_weights: torch.Tensor,
                              hunk_info: List[Dict],
                              save_name: str = "mil_attention") -> None:
        """Analyze MIL attention weights for hunk-level interpretability"""
        try:
            # Convert to numpy
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()

            # Create attention analysis
            attention_data = []
            for i, (weight, hunk) in enumerate(zip(attention_weights[0], hunk_info)):
                attention_data.append({
                    'hunk_id': i,
                    'attention_weight': float(weight[0]),
                    'file_path': hunk.get('file_path', 'unknown'),
                    'lines_added': len(hunk.get('added_lines', [])),
                    'lines_removed': len(hunk.get('removed_lines', []))
                })

            df = pd.DataFrame(attention_data)

            # Plot attention weights
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(df)), df['attention_weight'])
            plt.xlabel('Hunk Index')
            plt.ylabel('Attention Weight')
            plt.title('MIL Attention Weights per Hunk')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.save_path / f'{save_name}_bars.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Save attention data
            df.to_csv(self.save_path / f'{save_name}_data.csv', index=False)

            logger.info(f"MIL attention analysis saved as {save_name}")

        except Exception as e:
            logger.error(f"Error analyzing MIL attention: {e}")