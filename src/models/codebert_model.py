import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CodeBERTMultiTask(nn.Module):
    def __init__(self,
                 model_name: str = "microsoft/codebert-base",
                 num_quality_classes: int = 1,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.codebert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.codebert.config.hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Task-specific heads
        self.comment_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.config.vocab_size)
        )

        self.quality_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_quality_classes),
            nn.Sigmoid() if num_quality_classes == 1 else nn.Identity()
        )

        self.refinement_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.config.vocab_size)
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                task: str = 'all') -> Dict[str, torch.Tensor]:

        # Get CodeBERT embeddings
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Use pooler output for classification tasks
        pooled_output = self.dropout(outputs.pooler_output)
        sequence_output = outputs.last_hidden_state

        results = {}

        if task in ['comment', 'all']:
            # Generate review comments
            comment_logits = self.comment_generator(pooled_output)
            results['comment_logits'] = comment_logits

        if task in ['quality', 'all']:
            # Predict code quality
            quality_score = self.quality_predictor(pooled_output)
            results['quality_score'] = quality_score

        if task in ['refinement', 'all']:
            # Generate code refinements
            refinement_logits = self.refinement_head(pooled_output)
            results['refinement_logits'] = refinement_logits

        # Return attention weights for interpretability
        results['attention_weights'] = outputs.attentions[-1] if outputs.attentions else None
        results['hidden_states'] = sequence_output

        return results

    def generate_comment(self,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         tokenizer: AutoTokenizer,
                         max_length: int = 50) -> List[str]:
        """Generate review comments"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, task='comment')
            comment_logits = outputs['comment_logits']

            # Simple greedy decoding
            predicted_ids = torch.argmax(comment_logits, dim=-1)
            comments = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

        return comments

    def predict_quality(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict code quality scores"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, task='quality')
            return outputs['quality_score']