# src/features/enhanced_labeling.py

import re
import ast
from typing import Dict, List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def generate_realistic_labels(hunk_data: Dict) -> Tuple[float, bool, List[str]]:
    """Generate more realistic quality labels based on multiple heuristics"""
    added_lines = hunk_data['added_lines']
    context_lines = hunk_data.get('context_lines', [])

    code = '\n'.join(added_lines)
    score = 0.5  # Base score
    issues = []

    if not code.strip():
        return 0.0, False, ['empty_change']

    # Code complexity factors
    lines_count = len([l for l in added_lines if l.strip()])
    if lines_count > 50:
        score -= 0.2
        issues.append('very_long_change')
    elif lines_count > 20:
        score -= 0.1
        issues.append('long_change')

    # Code quality indicators
    comment_lines = len([l for l in added_lines if l.strip().startswith('#')])
    comment_ratio = comment_lines / lines_count if lines_count > 0 else 0

    if comment_ratio > 0.2:
        score += 0.1
    elif comment_ratio == 0 and lines_count > 10:
        score -= 0.1
        issues.append('no_comments')

    # Complexity patterns
    complexity_patterns = {
        r'\bif\b.*:\s*$': ('conditional', 0.05),
        r'\btry\b.*:\s*$': ('error_handling', 0.1),
        r'\bfor\b.*:\s*$': ('loop', 0.0),
        r'\bwhile\b.*:\s*$': ('while_loop', -0.05),
        r'\bdef\b.*:\s*$': ('function_def', 0.1),
        r'\bclass\b.*:\s*$': ('class_def', 0.15),
        r'TODO|FIXME|XXX': ('todo_markers', -0.1),
        r'print\(|console\.log': ('debug_prints', -0.15),
        r'import\s+\*': ('wildcard_import', -0.2),
    }

    for pattern, (name, weight) in complexity_patterns.items():
        matches = len(re.findall(pattern, code, re.MULTILINE))
        if matches > 0:
            score += weight * min(matches, 3)  # Cap at 3 occurrences
            if weight < 0:
                issues.append(f'{name}_detected')

    # Line length issues
    long_lines = [l for l in added_lines if len(l) > 100]
    if long_lines:
        score -= 0.05 * min(len(long_lines), 5)
        issues.append('long_lines')

    # Indentation consistency
    indents = [len(l) - len(l.lstrip()) for l in added_lines if l.strip()]
    if indents:
        indent_variety = len(set(indents))
        if indent_variety > 4:
            score -= 0.1
            issues.append('inconsistent_indentation')

    # AST-based quality checks
    try:
        # Try to parse the code fragment
        processed_code = preprocess_code_fragment(code)
        if processed_code:
            tree = ast.parse(processed_code)

            # Check for nested complexity
            max_nesting = calculate_nesting_depth(tree)
            if max_nesting > 4:
                score -= 0.1
                issues.append('deep_nesting')
            elif max_nesting > 2:
                score -= 0.05

            # Check for good practices
            has_docstrings = any(isinstance(node, ast.Expr) and
                                 isinstance(node.value, ast.Constant) and
                                 isinstance(node.value.value, str)
                                 for node in ast.walk(tree))
            if has_docstrings:
                score += 0.1

    except:
        # If can't parse, it's likely problematic
        score -= 0.2
        issues.append('unparseable_syntax')

    # Normalize score
    score = max(0.0, min(1.0, score))

    # More lenient binary classification threshold
    is_good_quality = score > 0.45  # Lowered from 0.6 to get more positive samples

    return score, is_good_quality, issues


def calculate_nesting_depth(tree) -> int:
    """Calculate maximum nesting depth of control structures"""

    class NestingVisitor(ast.NodeVisitor):
        def __init__(self):
            self.max_depth = 0
            self.current_depth = 0

        def visit_control_structure(self, node):
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
            self.generic_visit(node)
            self.current_depth -= 1

        def visit_If(self, node):
            self.visit_control_structure(node)

        def visit_For(self, node):
            self.visit_control_structure(node)

        def visit_While(self, node):
            self.visit_control_structure(node)

        def visit_With(self, node):
            self.visit_control_structure(node)

        def visit_Try(self, node):
            self.visit_control_structure(node)

    visitor = NestingVisitor()
    visitor.visit(tree)
    return visitor.max_depth


def extract_enhanced_features(hunk_data: Dict, commit_context: Dict = None) -> Dict:
    """Extract enhanced features with better discriminative power"""
    added_lines = hunk_data['added_lines']
    context_lines = hunk_data.get('context_lines', [])

    code = '\n'.join(added_lines)
    features = {}

    # Basic metrics
    features['lines_added'] = len([l for l in added_lines if l.strip()])
    features['total_chars'] = len(code)
    features['lines_removed'] = len(hunk_data.get('removed_lines', []))

    # Code structure features
    features['has_functions'] = 1 if re.search(r'\bdef\b', code) else 0
    features['has_classes'] = 1 if re.search(r'\bclass\b', code) else 0
    features['has_imports'] = 1 if re.search(r'^\s*(import|from)', code, re.MULTILINE) else 0

    # Quality indicators
    comment_lines = len([l for l in added_lines if l.strip().startswith('#')])
    features['comment_density'] = comment_lines / max(1, len(added_lines))
    features['avg_line_length'] = sum(len(l) for l in added_lines) / max(1, len(added_lines))
    features['max_line_length'] = max([len(l) for l in added_lines]) if added_lines else 0

    # Complexity features
    features['conditional_statements'] = len(re.findall(r'\b(if|elif)\b', code))
    features['loop_statements'] = len(re.findall(r'\b(for|while)\b', code))
    features['exception_handling'] = len(re.findall(r'\b(try|except|finally)\b', code))

    # Code smell indicators
    features['debug_prints'] = len(re.findall(r'print\s*\(', code))
    features['todo_comments'] = len(re.findall(r'TODO|FIXME|XXX', code, re.IGNORECASE))
    features['magic_numbers'] = len(re.findall(r'\b\d{2,}\b', code))  # Numbers with 2+ digits

    # Indentation features
    indents = [len(l) - len(l.lstrip()) for l in added_lines if l.strip()]
    if indents:
        features['max_indentation'] = max(indents)
        features['indent_variance'] = len(set(indents))
        features['avg_indentation'] = sum(indents) / len(indents)
    else:
        features['max_indentation'] = 0
        features['indent_variance'] = 0
        features['avg_indentation'] = 0

    # Context-based features
    if context_lines:
        context_code = '\n'.join(context_lines)
        features['context_similarity'] = calculate_text_similarity(code, context_code)
    else:
        features['context_similarity'] = 0

    # File-level context
    if commit_context:
        features['files_changed'] = len(commit_context.get('files', []))
        features['total_hunks'] = len(commit_context.get('hunks', []))
        features['commit_size'] = sum(len(h.get('added_lines', [])) for h in commit_context.get('hunks', []))
    else:
        features['files_changed'] = 1
        features['total_hunks'] = 1
        features['commit_size'] = features['lines_added']

    # Try AST features if possible
    try:
        processed_code = preprocess_code_fragment(code)
        if processed_code:
            tree = ast.parse(processed_code)
            features['nesting_depth'] = calculate_nesting_depth(tree)
            features['ast_parseable'] = 1

            # Count specific node types
            node_counts = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1

            features['unique_ast_nodes'] = len(node_counts)
            features['total_ast_nodes'] = sum(node_counts.values())
        else:
            features['nesting_depth'] = 0
            features['ast_parseable'] = 0
            features['unique_ast_nodes'] = 0
            features['total_ast_nodes'] = 0
    except:
        features['nesting_depth'] = 0
        features['ast_parseable'] = 0
        features['unique_ast_nodes'] = 0
        features['total_ast_nodes'] = 0

    return features


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity between two code snippets"""
    if not text1 or not text2:
        return 0.0

    # Simple word-based similarity
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def prepare_improved_baseline_features(commits_data: List[Dict]) -> Tuple[List[Dict], List[float], List[bool]]:
    """Prepare features with improved labeling and feature engineering"""
    all_features = []
    quality_scores = []
    binary_labels = []

    logger.info("Extracting improved features for baseline models...")

    for commit in commits_data:
        commit_context = {
            'files': commit.get('files_changed', []),
            'hunks': commit.get('hunks', [])
        }

        for hunk in commit['hunks']:
            if not any(line.strip() for line in hunk['added_lines']):
                continue

            try:
                # Extract enhanced features
                features = extract_enhanced_features(hunk, commit_context)

                # Generate realistic labels
                score, is_good, issues = generate_realistic_labels(hunk)

                # Add issue flags as features
                for issue in ['empty_change', 'very_long_change', 'long_change', 'no_comments',
                              'todo_markers', 'debug_prints', 'long_lines', 'deep_nesting',
                              'inconsistent_indentation', 'unparseable_syntax']:
                    features[f'has_{issue}'] = 1 if issue in issues else 0

                all_features.append(features)
                quality_scores.append(score)
                binary_labels.append(is_good)

            except Exception as e:
                logger.warning(f"Error processing hunk: {e}")
                continue

    return all_features, quality_scores, binary_labels


def analyze_label_distribution(commits_data: List[Dict], threshold: float = 0.45) -> Dict:
    """Analyze label distribution and provide recommendations"""
    scores = []
    issues_count = {}

    for commit in commits_data:
        for hunk in commit['hunks']:
            if not any(line.strip() for line in hunk['added_lines']):
                continue

            try:
                score, is_good, issues = generate_realistic_labels(hunk)
                scores.append(score)

                for issue in issues:
                    issues_count[issue] = issues_count.get(issue, 0) + 1
            except:
                continue

    if not scores:
        return {"error": "No valid samples found"}

    scores = np.array(scores)
    positive_count = np.sum(scores > threshold)

    analysis = {
        "total_samples": len(scores),
        "score_stats": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        },
        "threshold_analysis": {
            "current_threshold": threshold,
            "positive_samples": int(positive_count),
            "positive_ratio": float(positive_count / len(scores)),
            "recommended_threshold": float(np.percentile(scores, 70))  # 30% positive
        },
        "common_issues": dict(sorted(issues_count.items(), key=lambda x: x[1], reverse=True)[:5])
    }

    return analysis

def preprocess_code_fragment(code_fragment: str) -> str:
    """Preprocess code fragments to make them parseable"""
    lines = code_fragment.split('\n')
    lines = [line for line in lines if line.strip()]

    if not lines:
        return ""

    # Normalize indentation
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    if min_indent > 0:
        lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]

    processed_code = '\n'.join(lines)

    # Try parsing as-is
    try:
        ast.parse(processed_code)
        return processed_code
    except SyntaxError:
        pass

    # Try wrapping in function
    try:
        wrapped = f"def wrapper():\n" + '\n'.join(f"    {line}" for line in lines)
        ast.parse(wrapped)
        return wrapped
    except SyntaxError:
        pass

    return ""
