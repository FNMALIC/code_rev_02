import networkx as nx
import ast
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    def __init__(self):
        self.graph = nx.DiGraph()

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