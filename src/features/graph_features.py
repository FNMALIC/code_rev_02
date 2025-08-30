import networkx as nx
import ast
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    def __init__(self):
        self.graph = nx.DiGraph()

    def extract_features(self, code: str) -> Dict:
        """Extract graph-based features from code"""
        try:
            # Skip empty or whitespace-only code
            if not code or not code.strip():
                return self._empty_features()

            # Clean code and check for basic validity
            cleaned_code = self._clean_code(code)
            if not cleaned_code:
                return self._empty_features()

            tree = ast.parse(cleaned_code)

            # Build control flow graph
            cfg = self._build_control_flow_graph(tree)

            # Build dependency graph
            dep_graph = self._build_dependency_graph(tree)

            # Calculate graph metrics
            features = {
                'cfg_nodes': cfg.number_of_nodes(),
                'cfg_edges': cfg.number_of_edges(),
                'cfg_density': nx.density(cfg) if cfg.number_of_nodes() > 1 else 0,
                'cfg_connected_components': nx.number_connected_components(cfg.to_undirected()),
                'dep_nodes': dep_graph.number_of_nodes(),
                'dep_edges': dep_graph.number_of_edges(),
                'dep_density': nx.density(dep_graph) if dep_graph.number_of_nodes() > 1 else 0,
                'function_connectivity': self._calculate_function_connectivity(tree),
                'variable_scope_complexity': self._calculate_variable_scope_complexity(tree),
                'call_chain_depth': self._calculate_call_chain_depth(tree)
            }

            return features

        except (SyntaxError, ValueError, IndentationError) as e:
            logger.warning(f"Syntax error in code: {e}")
            return self._empty_features()
        except Exception as e:
            logger.warning(f"Error extracting graph features: {e}")
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

    def _build_control_flow_graph(self, tree) -> nx.DiGraph:
        """Build control flow graph from AST"""
        cfg = nx.DiGraph()

        class CFGBuilder(ast.NodeVisitor):
            def __init__(self, graph):
                self.graph = graph
                self.current_id = 0
                self.node_stack = []

            def get_node_id(self):
                self.current_id += 1
                return self.current_id

            def visit_FunctionDef(self, node):
                func_id = self.get_node_id()
                self.graph.add_node(func_id, type='function', name=node.name)
                self.node_stack.append(func_id)

                # Process function body
                for stmt in node.body:
                    self.visit(stmt)

                self.node_stack.pop()

            def visit_If(self, node):
                if_id = self.get_node_id()
                self.graph.add_node(if_id, type='if')

                if self.node_stack:
                    self.graph.add_edge(self.node_stack[-1], if_id)

                self.node_stack.append(if_id)

                # Process if body
                for stmt in node.body:
                    self.visit(stmt)

                # Process else body
                for stmt in node.orelse:
                    self.visit(stmt)

                self.node_stack.pop()

            def visit_For(self, node):
                loop_id = self.get_node_id()
                self.graph.add_node(loop_id, type='for')

                if self.node_stack:
                    self.graph.add_edge(self.node_stack[-1], loop_id)

                self.node_stack.append(loop_id)

                for stmt in node.body:
                    self.visit(stmt)

                self.node_stack.pop()

            def visit_While(self, node):
                loop_id = self.get_node_id()
                self.graph.add_node(loop_id, type='while')

                if self.node_stack:
                    self.graph.add_edge(self.node_stack[-1], loop_id)

                self.node_stack.append(loop_id)

                for stmt in node.body:
                    self.visit(stmt)

                self.node_stack.pop()

        builder = CFGBuilder(cfg)
        builder.visit(tree)

        return cfg

    def _build_dependency_graph(self, tree) -> nx.DiGraph:
        """Build dependency graph from variable usage"""
        dep_graph = nx.DiGraph()

        class DependencyBuilder(ast.NodeVisitor):
            def __init__(self, graph):
                self.graph = graph
                self.current_function = None
                self.variables = {}

            def visit_FunctionDef(self, node):
                self.current_function = node.name
                self.graph.add_node(node.name, type='function')
                self.generic_visit(node)

            def visit_Name(self, node):
                var_name = node.id

                if isinstance(node.ctx, ast.Store):
                    # Variable assignment
                    if var_name not in self.graph:
                        self.graph.add_node(var_name, type='variable')

                    if self.current_function:
                        self.graph.add_edge(self.current_function, var_name, type='defines')

                elif isinstance(node.ctx, ast.Load):
                    # Variable usage
                    if var_name not in self.graph:
                        self.graph.add_node(var_name, type='variable')

                    if self.current_function:
                        self.graph.add_edge(var_name, self.current_function, type='used_by')

                self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.graph:
                        self.graph.add_node(func_name, type='function')

                    if self.current_function and self.current_function != func_name:
                        self.graph.add_edge(self.current_function, func_name, type='calls')

                self.generic_visit(node)

        builder = DependencyBuilder(dep_graph)
        builder.visit(tree)

        return dep_graph

    def _calculate_function_connectivity(self, tree) -> int:
        """Calculate how interconnected functions are"""

        class FunctionAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.functions = set()
                self.calls = set()
                self.current_function = None

            def visit_FunctionDef(self, node):
                self.functions.add(node.name)
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = None

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and self.current_function:
                    func_name = node.func.id
                    if func_name in self.functions:
                        self.calls.add((self.current_function, func_name))
                self.generic_visit(node)

        analyzer = FunctionAnalyzer()
        analyzer.visit(tree)

        return len(analyzer.calls)

    def _calculate_variable_scope_complexity(self, tree) -> int:
        """Calculate complexity based on variable scoping"""

        class ScopeAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.scope_stack = [set()]  # Global scope
                self.complexity = 0

            def enter_scope(self):
                self.scope_stack.append(set())

            def exit_scope(self):
                if len(self.scope_stack) > 1:
                    scope_vars = self.scope_stack.pop()
                    self.complexity += len(scope_vars)

            def visit_FunctionDef(self, node):
                self.enter_scope()
                # Add parameters to scope
                for arg in node.args.args:
                    self.scope_stack[-1].add(arg.arg)
                self.generic_visit(node)
                self.exit_scope()

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.scope_stack[-1].add(node.id)
                self.generic_visit(node)

        analyzer = ScopeAnalyzer()
        analyzer.visit(tree)
        analyzer.exit_scope()  # Process global scope

        return analyzer.complexity

    def _calculate_call_chain_depth(self, tree) -> int:
        """Calculate maximum depth of function call chains"""

        class CallChainAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0

            def visit_Call(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

        analyzer = CallChainAnalyzer()
        analyzer.visit(tree)

        return analyzer.max_depth

    def _empty_features(self) -> Dict:
        """Return empty features when extraction fails"""
        return {
            'cfg_nodes': 0,
            'cfg_edges': 0,
            'cfg_density': 0,
            'cfg_connected_components': 0,
            'dep_nodes': 0,
            'dep_edges': 0,
            'dep_density': 0,
            'function_connectivity': 0,
            'variable_scope_complexity': 0,
            'call_chain_depth': 0
        }

    def create_dependency_graph(self, code: str) -> nx.DiGraph:
        """Create function/class dependency graph from code"""
        try:
            tree = ast.parse(code)
            graph = nx.DiGraph()

            class DependencyVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_function = None
                    self.functions = {}
                    self.calls = []

                def visit_FunctionDef(self, node):
                    self.current_function = node.name
                    self.functions[node.name] = node
                    graph.add_node(node.name, type='function')
                    self.generic_visit(node)
                    self.current_function = None

                def visit_ClassDef(self, node):
                    graph.add_node(node.name, type='class')
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = f"{node.name}.{item.name}"
                            graph.add_node(method_name, type='method')
                            graph.add_edge(node.name, method_name)
                    self.generic_visit(node)

                def visit_Call(self, node):
                    if self.current_function and isinstance(node.func, ast.Name):
                        called_function = node.func.id
                        if called_function in self.functions:
                            graph.add_edge(self.current_function, called_function)
                    self.generic_visit(node)

            visitor = DependencyVisitor()
            visitor.visit(tree)
            return graph

        except Exception as e:
            logger.warning(f"Error creating dependency graph: {e}")
            return nx.DiGraph()

    def extract_graph_features(self, graph: nx.DiGraph) -> Dict:
        """Extract graph-based features"""
        if graph.number_of_nodes() == 0:
            return self._empty_graph_features()

        features = {
            'basic_metrics': self._basic_metrics(graph),
            'centrality_measures': self._centrality_features(graph),
            'structural_metrics': self._structural_features(graph)
        }
        return features

    def _basic_metrics(self, graph: nx.DiGraph) -> Dict:
        """Basic graph metrics"""
        return {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_connected': nx.is_weakly_connected(graph)
        }

    def _centrality_features(self, graph: nx.DiGraph) -> Dict:
        """Calculate centrality measures"""
        try:
            centralities = {}

            # Degree centrality
            degree_cent = nx.degree_centrality(graph)
            centralities['avg_degree_centrality'] = sum(degree_cent.values()) / len(degree_cent) if degree_cent else 0
            centralities['max_degree_centrality'] = max(degree_cent.values()) if degree_cent else 0

            # Betweenness centrality
            between_cent = nx.betweenness_centrality(graph)
            centralities['avg_betweenness_centrality'] = sum(between_cent.values()) / len(
                between_cent) if between_cent else 0
            centralities['max_betweenness_centrality'] = max(between_cent.values()) if between_cent else 0

            # Closeness centrality
            close_cent = nx.closeness_centrality(graph)
            centralities['avg_closeness_centrality'] = sum(close_cent.values()) / len(close_cent) if close_cent else 0
            centralities['max_closeness_centrality'] = max(close_cent.values()) if close_cent else 0

            return centralities
        except:
            return {'avg_degree_centrality': 0, 'max_degree_centrality': 0,
                    'avg_betweenness_centrality': 0, 'max_betweenness_centrality': 0,
                    'avg_closeness_centrality': 0, 'max_closeness_centrality': 0}

    def _structural_features(self, graph: nx.DiGraph) -> Dict:
        """Calculate structural features"""
        try:
            features = {}

            # Clustering coefficient
            if graph.number_of_nodes() > 0:
                clustering = nx.clustering(graph.to_undirected())
                features['avg_clustering_coefficient'] = sum(clustering.values()) / len(clustering) if clustering else 0
                features['max_clustering_coefficient'] = max(clustering.values()) if clustering else 0
            else:
                features['avg_clustering_coefficient'] = 0
                features['max_clustering_coefficient'] = 0

            # Path length metrics
            if nx.is_weakly_connected(graph):
                try:
                    path_lengths = dict(nx.all_pairs_shortest_path_length(graph.to_undirected()))
                    all_lengths = [length for paths in path_lengths.values() for length in paths.values() if length > 0]
                    features['avg_path_length'] = sum(all_lengths) / len(all_lengths) if all_lengths else 0
                    features['diameter'] = max(all_lengths) if all_lengths else 0
                except:
                    features['avg_path_length'] = 0
                    features['diameter'] = 0
            else:
                features['avg_path_length'] = 0
                features['diameter'] = 0

            return features
        except:
            return {'avg_clustering_coefficient': 0, 'max_clustering_coefficient': 0,
                    'avg_path_length': 0, 'diameter': 0}

    def _empty_graph_features(self) -> Dict:
        """Return empty graph features"""
        return {
            'basic_metrics': {'num_nodes': 0, 'num_edges': 0, 'density': 0, 'is_connected': False},
            'centrality_measures': {'avg_degree_centrality': 0, 'max_degree_centrality': 0,
                                    'avg_betweenness_centrality': 0, 'max_betweenness_centrality': 0,
                                    'avg_closeness_centrality': 0, 'max_closeness_centrality': 0},
            'structural_metrics': {'avg_clustering_coefficient': 0, 'max_clustering_coefficient': 0,
                                   'avg_path_length': 0, 'diameter': 0}
        }