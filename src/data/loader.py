import git
import pandas as pd
from typing import List, Dict, Tuple, Optional
import ast
import json
import re
from pathlib import Path
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)


class CodeReviewDataLoader:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.temp_dir = None
        self.repo = None

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

    def extract_commits(self, limit: int = 1000,
                        file_extensions: List[str] = ['.py', '.html', '.css', '.js', '.json', '.md']) -> List[Dict]:
        """Extract commit data with diffs and metadata"""
        if not self.repo:
            logger.error(f"Repository not accessible")
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

                logger.info(f"Processing commit {commit.hexsha[:8]} - {commit.message.strip()[:50]}")

                # Extract file changes
                if commit.parents:
                    parent_commit = commit.parents[0]

                    # Try using git show command as alternative
                    try:
                        diff_text = self.repo.git.show(commit.hexsha, format='', no_merges=True)
                        logger.info(f"Git show diff length: {len(diff_text)}")

                        if diff_text:
                            # Parse the git show output
                            files_data = self.parse_git_show_output(diff_text, file_extensions)
                            commit_data['files_changed'].extend(files_data)

                            # Extract hunks from each file
                            for file_data in files_data:
                                hunks = self.extract_hunks(file_data['diff'])
                                logger.info(f"Extracted {len(hunks)} hunks from {file_data['file_path']}")
                                commit_data['hunks'].extend(hunks)

                    except Exception as e:
                        logger.warning(f"Error using git show: {e}")

                        # Fallback to original method
                        diffs = parent_commit.diff(commit)
                        logger.info(f"Found {len(diffs)} diffs")

                        for j, diff in enumerate(diffs):
                            if (diff.a_path and
                                    any(diff.a_path.endswith(ext) for ext in file_extensions)):

                                # Check if file is binary
                                is_binary = False
                                if diff.a_blob is not None:
                                    is_binary = getattr(diff.a_blob, 'binary', False)
                                elif diff.b_blob is not None:
                                    is_binary = getattr(diff.b_blob, 'binary', False)

                                if not is_binary:
                                    try:
                                        # Try to get diff using git command directly
                                        if diff.change_type == 'A':  # Added file
                                            diff_text = self.repo.git.show(f"{commit.hexsha}:{diff.b_path}")
                                            diff_text = f"--- /dev/null\n+++ b/{diff.b_path}\n" + '\n'.join(
                                                f'+{line}' for line in diff_text.split('\n'))
                                        elif diff.change_type == 'D':  # Deleted file
                                            diff_text = self.repo.git.show(f"{parent_commit.hexsha}:{diff.a_path}")
                                            diff_text = f"--- a/{diff.a_path}\n+++ /dev/null\n" + '\n'.join(
                                                f'-{line}' for line in diff_text.split('\n'))
                                        else:  # Modified file
                                            diff_text = self.repo.git.diff(parent_commit.hexsha, commit.hexsha,
                                                                           diff.a_path)

                                        if diff_text.strip():
                                            logger.info(f"File: {diff.a_path}, Diff length: {len(diff_text)}")

                                            file_data = {
                                                'file_path': diff.a_path,
                                                'change_type': diff.change_type,
                                                'diff': diff_text
                                            }

                                            commit_data['files_changed'].append(file_data)
                                            hunks = self.extract_hunks(diff_text)
                                            logger.info(f"Extracted {len(hunks)} hunks from {diff.a_path}")
                                            commit_data['hunks'].extend(hunks)
                                        else:
                                            logger.info(f"Empty diff for {diff.a_path}")

                                    except Exception as e:
                                        logger.warning(f"Error getting diff for {diff.a_path}: {e}")
                                        continue
                else:
                    logger.info("Commit has no parents (initial commit)")

                logger.info(
                    f"Commit {commit.hexsha[:8]} final stats: {len(commit_data['files_changed'])} files, {len(commit_data['hunks'])} hunks")
                commits_data.append(commit_data)

            except Exception as e:
                logger.warning(f"Error processing commit {commit.hexsha}: {e}")
                continue

        logger.info(f"Extracted {len(commits_data)} commits")
        return commits_data

    def parse_git_show_output(self, diff_text: str, file_extensions: List[str]) -> List[Dict]:
        """Parse git show output to extract file changes"""
        files_data = []
        current_file = None
        current_diff = []

        lines = diff_text.split('\n')

        for line in lines:
            # Check for file header
            if line.startswith('diff --git'):
                # Save previous file if exists
                if current_file and current_diff:
                    current_file['diff'] = '\n'.join(current_diff)
                    files_data.append(current_file)
                    current_diff = []

                # Extract file paths
                parts = line.split()
                if len(parts) >= 4:
                    a_path = parts[2][2:]  # Remove 'a/' prefix
                    b_path = parts[3][2:]  # Remove 'b/' prefix

                    # Use b_path for new files, a_path for others
                    file_path = b_path if b_path != '/dev/null' else a_path

                    if any(file_path.endswith(ext) for ext in file_extensions):
                        current_file = {
                            'file_path': file_path,
                            'change_type': 'M',  # Will be refined later
                            'diff': ''
                        }
                    else:
                        current_file = None

            # Collect diff content for current file
            if current_file:
                current_diff.append(line)

                # Determine change type from diff markers
                if line.startswith('new file mode'):
                    current_file['change_type'] = 'A'
                elif line.startswith('deleted file mode'):
                    current_file['change_type'] = 'D'

        # Add last file
        if current_file and current_diff:
            current_file['diff'] = '\n'.join(current_diff)
            files_data.append(current_file)

        return files_data
    def extract_hunks(self, diff_text: str) -> List[Dict]:
        """Extract individual hunks from diff text"""
        hunks = []
        hunk_pattern = r'@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@'

        logger.info(f"Processing diff text (first 200 chars): {diff_text[:200]}")

        lines = diff_text.split('\n')
        current_hunk = None

        for line in lines:
            hunk_match = re.match(hunk_pattern, line)
            if hunk_match:
                if current_hunk:
                    hunks.append(current_hunk)

                # Handle optional count values (default to 1 if not present)
                old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1

                current_hunk = {
                    'old_start': int(hunk_match.group(1)),
                    'old_count': old_count,
                    'new_start': int(hunk_match.group(3)),
                    'new_count': new_count,
                    'added_lines': [],
                    'removed_lines': [],
                    'context_lines': []
                }
                logger.info(f"Found hunk: -{hunk_match.group(1)},{old_count} +{hunk_match.group(3)},{new_count}")
            elif current_hunk:
                if line.startswith('+') and not line.startswith('+++'):
                    current_hunk['added_lines'].append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    current_hunk['removed_lines'].append(line[1:])
                elif line.startswith(' '):
                    current_hunk['context_lines'].append(line[1:])

        if current_hunk:
            hunks.append(current_hunk)

        logger.info(f"Extracted {len(hunks)} hunks")
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