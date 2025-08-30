import ast
import json
from typing import Dict, List, Optional
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class ASTFeatureExtractor:
    def __init__(self):
        self.node_types = set()

    def extract_features(self, code: str) -> Dict:
        """Extract comprehensive AST features - this is the missing method"""
        return self.extract_ast_features(code)

    def extract_ast_features(self, code: str) -> Dict:
        """Extract comprehensive AST features"""
        try:
            # Skip empty or whitespace-only code
            if not code or not code.strip():
                return self._empty_features()

            # Clean code and check for basic validity
            cleaned_code = self._clean_code(code)
            if not cleaned_code:
                return self._empty_features()

            tree = ast.parse(cleaned_code)

            # Get all the individual feature components
            node_counts = self._count_nodes(tree)
            depth = self._calculate_depth(tree)
            complexity_metrics = self._complexity_metrics(tree)
            structural_features = self._structural_features(tree)

            # Flatten all features into a single dict for compatibility
            features = {
                'ast_depth': depth,
                'num_functions': node_counts.get('FunctionDef', 0),
                'num_classes': node_counts.get('ClassDef', 0),
                'num_imports': node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0),
                'num_loops': node_counts.get('For', 0) + node_counts.get('While', 0),
                'num_conditionals': node_counts.get('If', 0),
                'num_try_blocks': node_counts.get('Try', 0),
                'cyclomatic_complexity': complexity_metrics['cyclomatic_complexity'],
                'cognitive_complexity': complexity_metrics['cognitive_complexity'],
                'function_calls': structural_features['function_calls'],
                'assignments': structural_features['assignments'],
                'returns': structural_features['returns'],
                'exceptions': structural_features['exceptions'],
                'total_nodes': sum(node_counts.values())
            }

            # Add individual node counts with prefix
            for node_type, count in node_counts.items():
                features[f'ast_{node_type.lower()}_count'] = count

            return features

        except (SyntaxError, ValueError, IndentationError) as e:
            logger.warning(f"Syntax error in code: {e}")
            return self._empty_features()
        except Exception as e:
            logger.warning(f"Error extracting AST features: {e}")
            return self._empty_features()

    def _clean_code(self, code: str) -> str:
        """Clean and prepare code for AST parsing"""
        try:
            lines = code.split('\n')
            cleaned_lines = []

            # Remove common issues that cause indent errors
            for line in lines:
                # Skip completely empty lines at start
                if not cleaned_lines and not line.strip():
                    continue

                # Handle lines with only whitespace
                if not line.strip():
                    cleaned_lines.append('')
                    continue

                cleaned_lines.append(line)

            if not cleaned_lines:
                return ''

            # Try to fix indentation by finding minimum indent
            non_empty_lines = [line for line in cleaned_lines if line.strip()]
            if not non_empty_lines:
                return ''

            min_indent = float('inf')
            for line in non_empty_lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    if indent < min_indent:
                        min_indent = indent

            # If all lines are indented, dedent them
            if min_indent > 0 and min_indent != float('inf'):
                dedented_lines = []
                for line in cleaned_lines:
                    if line.strip():  # Non-empty line
                        dedented_lines.append(line[min_indent:])
                    else:  # Empty line
                        dedented_lines.append('')
                cleaned_lines = dedented_lines

            cleaned_code = '\n'.join(cleaned_lines)

            # Test if it parses
            try:
                ast.parse(cleaned_code)
                return cleaned_code
            except:
                # If still fails, try wrapping in a function
                wrapped_code = f"def temp_function():\n"
                for line in cleaned_lines:
                    wrapped_code += f"    {line}\n"
                try:
                    ast.parse(wrapped_code)
                    return wrapped_code
                except:
                    return ''

        except Exception as e:
            logger.warning(f"Error cleaning code: {e}")
            return ''

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
            for child in ast.iter_child_nodes(node):
                child_depth = _depth(child, current_depth + 1)
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

            def visit_With(self, node):
                self.cyclomatic_complexity += 1
                self.cognitive_complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_ExceptHandler(self, node):
                self.cyclomatic_complexity += 1
                self.generic_visit(node)

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
                self.variables = set()
                self.functions = []

            def visit_Call(self, node):
                self.function_calls += 1
                self.generic_visit(node)

            def visit_Assign(self, node):
                self.assignments += 1
                # Track variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.variables.add(target.id)
                self.generic_visit(node)

            def visit_Return(self, node):
                self.returns += 1
                self.generic_visit(node)

            def visit_Raise(self, node):
                self.exceptions += 1
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                self.functions.append(node.name)
                self.generic_visit(node)

        visitor = StructuralVisitor()
        visitor.visit(tree)

        return {
            'function_calls': visitor.function_calls,
            'assignments': visitor.assignments,
            'returns': visitor.returns,
            'exceptions': visitor.exceptions,
            'unique_variables': len(visitor.variables),
            'function_names': visitor.functions
        }

    def _empty_features(self) -> Dict:
        """Return empty features dict when parsing fails"""
        return {
            'ast_depth': 0,
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'num_loops': 0,
            'num_conditionals': 0,
            'num_try_blocks': 0,
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'function_calls': 0,
            'assignments': 0,
            'returns': 0,
            'exceptions': 0,
            'total_nodes': 0,
            'unique_variables': 0,
            'function_names': []
        }