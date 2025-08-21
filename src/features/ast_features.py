import ast
import json
from typing import Dict, List, Optional
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class ASTFeatureExtractor:
    def __init__(self):
        self.node_types = set()

    def extract_ast_features(self, code: str) -> Dict:
        """Extract comprehensive AST features"""
        try:
            tree = ast.parse(code)
            features = {
                'node_counts': self._count_nodes(tree),
                'depth': self._calculate_depth(tree),
                'complexity_metrics': self._complexity_metrics(tree),
                'structural_features': self._structural_features(tree)
            }
            return features
        except Exception as e:
            logger.warning(f"Error extracting AST features: {e}")
            return self._empty_features()

    def _count_nodes(self, tree) -> Dict:
        """Count different AST node types"""
        node_counts = {}
        for node in ast.walk(tree):
            node_type = type(node).__name__
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
            self.node_types.add(node_type)
        return node_counts

    def _calculate_depth(self, tree) -> int:
        """Calculate maximum AST depth"""

        def _depth(node, current_depth=0):
            if not hasattr(node, '_fields'):
                return current_depth

            max_child_depth = current_depth
            for field, value in ast.iter_child_nodes(node):
                child_depth = _depth(value, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth

        return _depth(tree)

    def _complexity_metrics(self, tree) -> Dict:
        """Calculate complexity metrics"""

        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.cyclomatic_complexity = 1  # Base complexity
                self.cognitive_complexity = 0
                self.nesting_level = 0

            def visit_If(self, node):
                self.cyclomatic_complexity += 1
                self.cognitive_complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_For(self, node):
                self.cyclomatic_complexity += 1
                self.cognitive_complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_While(self, node):
                self.cyclomatic_complexity += 1
                self.cognitive_complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_Try(self, node):
                self.cyclomatic_complexity += 1
                self.cognitive_complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        return {
            'cyclomatic_complexity': visitor.cyclomatic_complexity,
            'cognitive_complexity': visitor.cognitive_complexity
        }

    def _structural_features(self, tree) -> Dict:
        """Extract structural features"""

        class StructuralVisitor(ast.NodeVisitor):
            def __init__(self):
                self.function_calls = 0
                self.assignments = 0
                self.returns = 0
                self.exceptions = 0

            def visit_Call(self, node):
                self.function_calls += 1
                self.generic_visit(node)

            def visit_Assign(self, node):
                self.assignments += 1
                self.generic_visit(node)

            def visit_Return(self, node):
                self.returns += 1
                self.generic_visit(node)

            def visit_Raise(self, node):
                self.exceptions += 1
                self.generic_visit(node)

        visitor = StructuralVisitor()
        visitor.visit(tree)

        return {
            'function_calls': visitor.function_calls,
            'assignments': visitor.assignments,
            'returns': visitor.returns,
            'exceptions': visitor.exceptions
        }

    def _empty_features(self) -> Dict:
        """Return empty features dict when parsing fails"""
        return {
            'node_counts': {},
            'depth': 0,
            'complexity_metrics': {'cyclomatic_complexity': 0, 'cognitive_complexity': 0},
            'structural_features': {'function_calls': 0, 'assignments': 0, 'returns': 0, 'exceptions': 0}
        }