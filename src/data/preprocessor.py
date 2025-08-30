import re
import ast
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CodePreprocessor:
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512

    def preprocess_code(self, code: str) -> str:  # <-- ADDED: This is the method that main.py was looking for
        """Main preprocessing method that the main script expects"""
        return self.clean_code(code)

    def clean_code(self, code: str) -> str:
        """Clean and normalize code"""
        code = re.sub(r'\n\s*\n', '\n', code)
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        return code.strip()

    def tokenize_code(self, code: str) -> Dict:
        """Tokenize code for model input"""
        cleaned_code = self.clean_code(code)
        return self.tokenizer(
            cleaned_code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

    def extract_code_metrics(self, code: str) -> Dict:
        """Extract basic code metrics"""
        lines = code.split('\n')
        metrics = {
            'lines_of_code': len([l for l in lines if l.strip()]),
            'total_lines': len(lines),
            'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
            'blank_lines': len([l for l in lines if not l.strip()]),
            'max_line_length': max([len(l) for l in lines]) if lines else 0,
            'avg_line_length': sum([len(l) for l in lines]) / len(lines) if lines else 0
        }

        try:
            tree = ast.parse(code)
            metrics.update(self._ast_metrics(tree))
        except:
            logger.warning("Could not parse code for AST metrics")

        return metrics

    def _ast_metrics(self, tree) -> Dict:
        """Extract AST-based metrics"""
        class MetricsVisitor(ast.NodeVisitor):
            def __init__(self):
                self.functions = 0
                self.classes = 0
                self.imports = 0
                self.max_depth = 0
                self.current_depth = 0

            def visit_FunctionDef(self, node):
                self.functions += 1
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

            def visit_ClassDef(self, node):
                self.classes += 1
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

            def visit_Import(self, node):
                self.imports += 1
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                self.imports += 1
                self.generic_visit(node)

        visitor = MetricsVisitor()
        visitor.visit(tree)

        return {
            'num_functions': visitor.functions,
            'num_classes': visitor.classes,
            'num_imports': visitor.imports,
            'max_nesting_depth': visitor.max_depth
        }