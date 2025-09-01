import re
from typing import Dict, List, Tuple, Optional


class GitDiffParser:
    """Parse git diff output into structured data"""

    def __init__(self):
        self.file_header_pattern = re.compile(r'^diff --git a/(.*?) b/(.*?)$')
        self.hunk_header_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

    def parse_diff(self, diff_output: str) -> List[Dict]:
        """Parse git diff output into structured hunks"""
        if not diff_output:
            return []

        lines = diff_output.split('\n')
        files = []
        current_file = None
        current_hunk = None

        for line in lines:
            # Check for file header
            file_match = self.file_header_pattern.match(line)
            if file_match:
                if current_file:
                    files.append(current_file)

                current_file = {
                    'file_path': file_match.group(2),
                    'old_file': file_match.group(1),
                    'new_file': file_match.group(2),
                    'hunks': []
                }
                current_hunk = None
                continue

            # Check for hunk header
            hunk_match = self.hunk_header_pattern.match(line)
            if hunk_match and current_file:
                if current_hunk:
                    current_file['hunks'].append(current_hunk)

                old_start = int(hunk_match.group(1))
                old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                new_start = int(hunk_match.group(3))
                new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1

                current_hunk = {
                    'old_start': old_start,
                    'old_count': old_count,
                    'new_start': new_start,
                    'new_count': new_count,
                    'added_lines': [],
                    'removed_lines': [],
                    'context_lines': [],
                    'line_start': new_start,
                    'line_end': new_start + new_count - 1,
                    'raw_lines': []
                }
                continue

            # Process hunk content
            if current_hunk is not None:
                current_hunk['raw_lines'].append(line)

                if line.startswith('+') and not line.startswith('+++'):
                    current_hunk['added_lines'].append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    current_hunk['removed_lines'].append(line[1:])
                elif line.startswith(' '):
                    current_hunk['context_lines'].append(line[1:])

        # Add final file and hunk
        if current_hunk and current_file:
            current_file['hunks'].append(current_hunk)
        if current_file:
            files.append(current_file)


        return files


class EnhancedGitDiffParser(GitDiffParser):
    def parse_diff(self, diff_output: str) -> List[Dict]:
        files = super().parse_diff(diff_output)

        # Process files that had no hunks but have changes
        for file_data in files:
            if not file_data['hunks']:
                # Check if it's a new file or has binary changes
                if 'new file mode' in diff_output or 'Binary files' in diff_output:
                    file_data['hunks'].append({
                        'type': 'new_file',
                        'added_lines': ['[New file or binary change]'],
                        'removed_lines': [],
                        'context_lines': [],
                        'line_start': 1,
                        'line_end': 1
                    })

        return files