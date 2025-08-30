import git
from typing import Dict, List, Optional, Tuple
import json
import subprocess
from pathlib import Path
import logging
import tempfile
import shutil
from .diff_parser import GitDiffParser
logger = logging.getLogger(__name__)


class GitIntegration:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.temp_dir = None
        self.repo = None
        self.diff_parser = GitDiffParser()

        # Check if it's a URL or local path
        if repo_path.startswith(('http://', 'https://', 'git@')):
            # Clone the repository to a temporary directory
            self.temp_dir = tempfile.mkdtemp()
            try:
                logger.info(f"Cloning repository from {repo_path} to {self.temp_dir}")
                self.repo = git.Repo.clone_from(repo_path, self.temp_dir)
                self.repo_path = Path(self.temp_dir)
            except Exception as e:
                logger.error(f"Failed to clone repository: {e}")
                self.repo = None
        else:
            # Local repository
            self.repo_path = Path(repo_path)
            if self.repo_path.exists():
                try:
                    self.repo = git.Repo(repo_path)
                except Exception as e:
                    logger.error(f"Failed to open repository: {e}")
                    self.repo = None
            else:
                logger.error(f"Repository not found at {self.repo_path}")
                self.repo = None

    def __del__(self):
        """Cleanup temporary directory if created"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def get_current_diff(self) -> str:
        """Get unstaged changes"""
        if not self.repo:
            return ""
        return self.repo.git.diff()

    def get_staged_diff(self) -> str:
        """Get staged changes"""
        if not self.repo:
            return ""
        return self.repo.git.diff('--cached')

    def analyze_commit(self, commit_hash: str) -> Dict:
        """Extract and analyze specific commit"""
        if not self.repo:
            return {}

        try:
            commit = self.repo.commit(commit_hash)

            commit_data = {
                'hash': commit.hexsha,
                'message': commit.message.strip(),
                'author': str(commit.author),
                'date': commit.committed_datetime.isoformat(),
                'files_changed': [],
                'stats': {
                    'total_insertions': 0,
                    'total_deletions': 0,
                    'files_changed': 0
                }
            }

            # Get commit stats
            stats = commit.stats
            commit_data['stats']['total_insertions'] = stats.total['insertions']
            commit_data['stats']['total_deletions'] = stats.total['deletions']
            commit_data['stats']['files_changed'] = stats.total['files']

            # Get file changes
            if commit.parents:
                diffs = commit.parents[0].diff(commit)
                for diff in diffs:
                    if diff.a_path:
                        file_data = {
                            'path': diff.a_path,
                            'change_type': diff.change_type,
                            'insertions': diff.change_type != 'D' and len(
                                diff.diff.decode('utf-8', errors='ignore').split('\n+')) - 1,
                            'deletions': diff.change_type != 'A' and len(
                                diff.diff.decode('utf-8', errors='ignore').split('\n-')) - 1
                        }
                        commit_data['files_changed'].append(file_data)

            return commit_data

        except Exception as e:
            logger.error(f"Error analyzing commit {commit_hash}: {e}")
            return {}

    def create_review_comment(self,
                              file_path: str,
                              line_number: int,
                              comment: str,
                              commit_hash: Optional[str] = None) -> bool:
        """Create review comment (placeholder for PR/MR integration)"""
        try:
            review_data = {
                'file_path': file_path,
                'line_number': line_number,
                'comment': comment,
                'commit_hash': commit_hash or 'current',
            }

            # Save to review file (in real implementation, this would integrate with GitHub/GitLab API)
            review_file = self.repo_path / '.git' / 'code_reviews.json'

            if review_file.exists():
                with open(review_file, 'r') as f:
                    reviews = json.load(f)
            else:
                reviews = []

            reviews.append(review_data)

            with open(review_file, 'w') as f:
                json.dump(reviews, f, indent=2)

            logger.info(f"Review comment created for {file_path}:{line_number}")
            return True

        except Exception as e:
            logger.error(f"Error creating review comment: {e}")
            return False

    def setup_pre_commit_hook(self, script_path: str) -> bool:
        """Set up pre-commit hook for automated review"""
        try:
            hook_dir = self.repo_path / '.git' / 'hooks'
            hook_file = hook_dir / 'pre-commit'

            hook_content = f"""#!/bin/bash
# Automated code review hook
python {script_path} --hook-mode
exit $?
"""

            with open(hook_file, 'w') as f:
                f.write(hook_content)

            # Make executable
            subprocess.run(['chmod', '+x', str(hook_file)], check=True)

            logger.info("Pre-commit hook installed successfully")
            return True

        except Exception as e:
            logger.error(f"Error setting up pre-commit hook: {e}")
            return False

    def get_file_at_commit(self, file_path: str, commit_hash: str) -> str:
        """Get file content at specific commit"""
        try:
            commit = self.repo.commit(commit_hash)
            return commit.tree[file_path].data_stream.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error getting file {file_path} at commit {commit_hash}: {e}")
            return ""

    def get_recent_commits(self, limit: int = 10, file_path: Optional[str] = None) -> List[Dict]:
        """Get recent commits with optional file filtering"""
        if not self.repo:
            return []

        commits = []
        commit_iter = self.repo.iter_commits(paths=file_path) if file_path else self.repo.iter_commits()

        for i, commit in enumerate(commit_iter):
            if i >= limit:
                break

            commit_info = {
                'hash': commit.hexsha,
                'short_hash': commit.hexsha[:8],
                'message': commit.message.strip(),
                'author': str(commit.author),
                'date': commit.committed_datetime.isoformat()
            }
            commits.append(commit_info)

        return commits