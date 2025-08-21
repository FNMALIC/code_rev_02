from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ModelEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def evaluate_classification(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate classification performance"""

        # Convert probabilities to binary predictions if needed
        if y_prob is not None and len(np.unique(y_pred)) == 1:
            y_pred = (y_prob > 0.5).astype(int)

        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # AUC if probabilities available
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                    metrics['auc'] = roc_auc_score(y_true, y_prob)
                else:
                    metrics['auc'] = 0.0
            except ValueError:
                metrics['auc'] = 0.0

        return metrics

    def evaluate_generation(self,
                            generated: List[str],
                            reference: List[str]) -> Dict[str, float]:
        """Evaluate text generation quality"""
        if len(generated) != len(reference):
            logger.warning("Generated and reference lists have different lengths")
            min_len = min(len(generated), len(reference))
            generated = generated[:min_len]
            reference = reference[:min_len]

        # BLEU scores
        bleu_scores = []
        for gen, ref in zip(generated, reference):
            try:
                ref_tokens = [nltk.word_tokenize(ref.lower())]
                gen_tokens = nltk.word_tokenize(gen.lower())
                bleu = sentence_bleu(ref_tokens, gen_tokens)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)

        # ROUGE scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for gen, ref in zip(generated, reference):
            try:
                scores = self.rouge_scorer.score(ref, gen)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            except:
                rouge_scores['rouge1'].append(0.0)
                rouge_scores['rouge2'].append(0.0)
                rouge_scores['rougeL'].append(0.0)

        metrics = {
            'bleu': np.mean(bleu_scores),
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }

        return metrics

    def evaluate_multi_task(self,
                            results: Dict[str, Tuple]) -> Dict[str, Dict[str, float]]:
        """Evaluate multi-task model performance"""
        evaluation_results = {}

        for task, (y_true, y_pred, y_prob) in results.items():
            if task in ['quality', 'classification']:
                evaluation_results[task] = self.evaluate_classification(y_true, y_pred, y_prob)
            elif task in ['comment', 'refinement', 'generation']:
                evaluation_results[task] = self.evaluate_generation(y_pred, y_true)

        return evaluation_results

    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Create confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report"""
        return classification_report(y_true, y_pred)