

# TODO: DON'T TOUCH
# import torch


# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Tuple, Dict
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# class MultiInstanceLearning(nn.Module):
#     def __init__(self,
#                  input_dim: int,
#                  hidden_dim: int = 256,
#                  attention_dim: int = 128,
#                  num_classes: int = 1):
#         super().__init__()
#
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.attention_dim = attention_dim
#         self.num_classes = num_classes
#
#         # Instance-level feature extraction
#         self.instance_encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
#
#         # Attention mechanism for instance aggregation
#         self.attention_net = nn.Sequential(
#             nn.Linear(hidden_dim, attention_dim),
#             nn.Tanh(),
#             nn.Linear(attention_dim, 1)
#         )
#
#         # Bag-level classifier
#         self.bag_classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim // 2, num_classes)
#         )
#
#         if num_classes == 1:
#             self.activation = nn.Sigmoid()
#         else:
#             self.activation = nn.Softmax(dim=1)
#
#     def forward(self, instances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass for multi-instance learning
#
#         Args:
#             instances: Tensor of shape [batch_size, num_instances, input_dim]
#
#         Returns:
#             predictions: Bag-level predictions [batch_size, num_classes]
#             attention_weights: Instance attention weights [batch_size, num_instances, 1]
#         """
#         batch_size, num_instances, _ = instances.shape
#
#         # Reshape for batch processing
#         instances_flat = instances.view(-1, self.input_dim)
#
#         # Encode instances
#         encoded_instances = self.instance_encoder(instances_flat)
#         encoded_instances = encoded_instances.view(batch_size, num_instances, self.hidden_dim)
#
#         # Compute attention weights
#         attention_logits = self.attention_net(encoded_instances)  # [batch_size, num_instances, 1]
#         attention_weights = F.softmax(attention_logits, dim=1)
#
#         # Aggregate instances using attention
#         bag_representation = torch.sum(attention_weights * encoded_instances, dim=1)  # [batch_size, hidden_dim]
#
#         # Classify bags
#         logits = self.bag_classifier(bag_representation)
#         predictions = self.activation(logits)
#
#         return predictions, attention_weights
#
#     def predict_instances(self, instances: torch.Tensor) -> torch.Tensor:
#         """Predict instance-level scores"""
#         batch_size, num_instances, _ = instances.shape
#         instances_flat = instances.view(-1, self.input_dim)
#
#         # Get instance encodings
#         encoded_instances = self.instance_encoder(instances_flat)
#
#         # Instance-level predictions
#         instance_logits = self.bag_classifier(encoded_instances)
#         instance_predictions = self.activation(instance_logits)
#
#         return instance_predictions.view(batch_size, num_instances, self.num_classes)
#
#
# class CodeBERTMILIntegrated(nn.Module):
#     """Integration of CodeBERT with Multi-Instance Learning for hunk-level analysis"""
#
#     def __init__(self,
#                  codebert_model: nn.Module,
#                  mil_input_dim: int = 768,
#                  mil_hidden_dim: int = 256):
#         super().__init__()
#
#         self.codebert = codebert_model
#         self.mil_model = MultiInstanceLearning(
#             input_dim=mil_input_dim,
#             hidden_dim=mil_hidden_dim
#         )
#
#         # Feature projection layer
#         self.feature_projection = nn.Linear(
#             self.codebert.hidden_size,
#             mil_input_dim
#         )
#
#     def forward(self,
#                 batch_input_ids: List[torch.Tensor],
#                 batch_attention_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """
#         Forward pass for CodeBERT-MIL integration
#
#         Args:
#             batch_input_ids: List of tensors, each [num_hunks, seq_len]
#             batch_attention_masks: List of tensors, each [num_hunks, seq_len]
#         """
#         batch_size = len(batch_input_ids)
#         batch_features = []
#
#         # Process each commit's hunks through CodeBERT
#         for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
#             if input_ids.size(0) == 0:  # No hunks
#                 # Create dummy features
#                 hunk_features = torch.zeros(1, self.codebert.hidden_size)
#             else:
#                 # Get CodeBERT features for all hunks
#                 with torch.no_grad():
#                     outputs = self.codebert.codebert(
#                         input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         return_dict=True
#                     )
#                     hunk_features = outputs.pooler_output  # [num_hunks, hidden_size]
#
#             # Project features to MIL input dimension
#             projected_features = self.feature_projection(hunk_features)
#             batch_features.append(projected_features)
#
#         # Pad sequences to same length
#         max_hunks = max([f.size(0) for f in batch_features])
#         padded_features = []
#
#         for features in batch_features:
#             num_hunks = features.size(0)
#             if num_hunks < max_hunks:
#                 padding = torch.zeros(max_hunks - num_hunks, features.size(1))
#                 features = torch.cat([features, padding], dim=0)
#             padded_features.append(features)
#
#         # Stack into tensor [batch_size, max_hunks, feature_dim]
#         instance_features = torch.stack(padded_features, dim=0)
#
#         # Apply MIL
#         bag_predictions, attention_weights = self.mil_model(instance_features)
#
#         return {
#             'bag_predictions': bag_predictions,
#             'attention_weights': attention_weights,
#             'instance_features': instance_features
#         }






import pytest
import torch
import numpy as np
from src.models.codebert_model import CodeBERTMultiTask
from src.models.baselines import BaselineModels
from src.models.mil_model import MultiInstanceLearning


class TestCodeBERTModel:
    def test_model_initialization(self):
        model = CodeBERTMultiTask()
        assert model is not None
        assert hasattr(model, 'codebert')
        assert hasattr(model, 'quality_predictor')

    def test_forward_pass(self):
        model = CodeBERTMultiTask()
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)

        outputs = model(input_ids, attention_mask)
        assert 'quality_score' in outputs
        assert outputs['quality_score'].shape == (2, 1)

    def test_single_task_forward(self):
        model = CodeBERTMultiTask()
        input_ids = torch.randint(0, 1000, (1, 128))
        attention_mask = torch.ones(1, 128)

        outputs = model(input_ids, attention_mask, task='quality')
        assert 'quality_score' in outputs
        assert 'comment_logits' not in outputs


class TestBaselineModels:
    def test_feature_preparation(self):
        baseline = BaselineModels()
        feature_dicts = [
            {'feature1': 1, 'feature2': 2},
            {'feature1': 3, 'feature2': 4}
        ]

        X = baseline.prepare_features(feature_dicts)
        assert X.shape == (2, 2)

    def test_model_training(self):
        baseline = BaselineModels()
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        models = baseline.train_all_models(X, y)
        assert 'logistic' in models
        assert 'random_forest' in models
        assert 'svm' in models


class TestMILModel:
    def test_mil_initialization(self):
        model = MultiInstanceLearning(input_dim=100)
        assert model.input_dim == 100
        assert model.hidden_dim == 256

    def test_mil_forward(self):
        model = MultiInstanceLearning(input_dim=50)
        instances = torch.randn(2, 10, 50)  # batch_size=2, num_instances=10

        predictions, attention = model(instances)
        assert predictions.shape == (2, 1)
        assert attention.shape == (2, 10, 1)