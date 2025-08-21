import git
import pandas as pd
from typing import List, Dict, Tuple, Optional
import ast
import json
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CodeReviewDataLoader:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path) if self.repo_path.exists() else None

    def extract_commits(self, limit: int = 1000, file_extensions: List[str] = ['.py']) -> List[Dict]:
        """Extract commit data with diffs and metadata"""
        if not self.repo:
            logger.error(f"Repository not found at {self.repo_path}")
            return []

        commits_data = []
        logger.info(f"Extracting {limit} commits...")

        for i, commit in enumerate(self.repo.iter_commits()):
            if i >= limit:
                break

            try:
                commit_data = {
                    'hash': commit.hexsha,
                    'message': commit.message.strip(),
                    'author': str(commit.author),
                    'date': commit.committed_datetime,
                    'files_changed': [],
                    'hunks': []
                }

                # Extract file changes
                if commit.parents:
                    diffs = commit.parents[0].diff(commit)
                    for diff in diffs:
                        if diff.a_path and any(diff.a_path.endswith(ext) for ext in file_extensions):
                            file_data = {
                                'file_path': diff.a_path,
                                'change_type': diff.change_type,
                                'diff': str(diff.diff.decode('utf-8', errors='ignore'))
                            }
                            commit_data['files_changed'].append(file_data)
                            commit_data['hunks'].extend(self.extract_hunks(file_data['diff']))

                commits_data.append(commit_data)

            except Exception as e:
                logger.warning(f"Error processing commit {commit.hexsha}: {e}")
                continue

        logger.info(f"Extracted {len(commits_data)} commits")
        return commits_data

    def extract_hunks(self, diff_text: str) -> List[Dict]:
        """Extract individual hunks from diff text"""
        hunks = []
        hunk_pattern = r'@@\s*-(\d+),(\d+)\s*\+(\d+),(\d+)\s*@@'

        lines = diff_text.split('\n')
        current_hunk = None

        for line in lines:
            hunk_match = re.match(hunk_pattern, line)
            if hunk_match:
                if current_hunk:
                    hunks.append(current_hunk)

                current_hunk = {
                    'old_start': int(hunk_match.group(1)),
                    'old_count': int(hunk_match.group(2)),
                    'new_start': int(hunk_match.group(3)),
                    'new_count': int(hunk_match.group(4)),
                    'added_lines': [],
                    'removed_lines': [],
                    'context_lines': []
                }
            elif current_hunk:
                if line.startswith('+') and not line.startswith('+++'):
                    current_hunk['added_lines'].append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    current_hunk['removed_lines'].append(line[1:])
                elif line.startswith(' '):
                    current_hunk['context_lines'].append(line[1:])

        if current_hunk:
            hunks.append(current_hunk)

        return hunks

    def load_review_comments(self, comments_file: Optional[str] = None) -> pd.DataFrame:
        """Load existing review comments if available"""
        if comments_file and Path(comments_file).exists():
            return pd.read_csv(comments_file)
        else:
            # Generate synthetic review data for training
            return self.generate_synthetic_reviews()

    def generate_synthetic_reviews(self) -> pd.DataFrame:
        """Generate synthetic review data for testing"""
        synthetic_data = {
            'commit_hash': ['abc123', 'def456', 'ghi789'],
            'file_path': ['main.py', 'utils.py', 'model.py'],
            'line_number': [10, 25, 50],
            'comment': ['Consider using list comprehension', 'Add error handling', 'Optimize this loop'],
            'quality_score': [0.7, 0.8, 0.6]
        }
        return pd.DataFrame(synthetic_data)