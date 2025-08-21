import torch
import torch.nn as nn
from typing import List, Tuple

# TODO: refth
class MultiInstanceLearning(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, instances):
        # instances: [batch_size, num_hunks, feature_dim]
        attention_weights = torch.softmax(self.attention(instances), dim=1)
        weighted_features = torch.sum(attention_weights * instances, dim=1)
        predictions = self.classifier(weighted_features)
        return predictions, attention_weights