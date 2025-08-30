#
#
# # TODO: DON'T TOUCH
# # import torch
#
#
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from typing import List, Tuple, Dict
# # import logging
# #
# # logger = logging.getLogger(__name__)
# #
# #
# # class MultiInstanceLearning(nn.Module):
# #     def __init__(self,
# #                  input_dim: int,
# #                  hidden_dim: int = 256,
# #                  attention_dim: int = 128,
# #                  num_classes: int = 1):
# #         super().__init__()
# #
# #         self.input_dim = input_dim
# #         self.hidden_dim = hidden_dim
# #         self.attention_dim = attention_dim
# #         self.num_classes = num_classes
# #
# #         # Instance-level feature extraction
# #         self.instance_encoder = nn.Sequential(
# #             nn.Linear(input_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.ReLU()
# #         )
# #
# #         # Attention mechanism for instance aggregation
# #         self.attention_net = nn.Sequential(
# #             nn.Linear(hidden_dim, attention_dim),
# #             nn.Tanh(),
# #             nn.Linear(attention_dim, 1)
# #         )
# #
# #         # Bag-level classifier
# #         self.bag_classifier = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim // 2),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(hidden_dim // 2, num_classes)
# #         )
# #
# #         if num_classes == 1:
# #             self.activation = nn.Sigmoid()
# #         else:
# #             self.activation = nn.Softmax(dim=1)
# #
# #     def forward(self, instances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# #         """
# #         Forward pass for multi-instance learning
# #
# #         Args:
# #             instances: Tensor of shape [batch_size, num_instances, input_dim]
# #
# #         Returns:
# #             predictions: Bag-level predictions [batch_size, num_classes]
# #             attention_weights: Instance attention weights [batch_size, num_instances, 1]
# #         """
# #         batch_size, num_instances, _ = instances.shape
# #
# #         # Reshape for batch processing
# #         instances_flat = instances.view(-1, self.input_dim)
# #
# #         # Encode instances
# #         encoded_instances = self.instance_encoder(instances_flat)
# #         encoded_instances = encoded_instances.view(batch_size, num_instances, self.hidden_dim)
# #
# #         # Compute attention weights
# #         attention_logits = self.attention_net(encoded_instances)  # [batch_size, num_instances, 1]
# #         attention_weights = F.softmax(attention_logits, dim=1)
# #
# #         # Aggregate instances using attention
# #         bag_representation = torch.sum(attention_weights * encoded_instances, dim=1)  # [batch_size, hidden_dim]
# #
# #         # Classify bags
# #         logits = self.bag_classifier(bag_representation)
# #         predictions = self.activation(logits)
# #
# #         return predictions, attention_weights
# #
# #     def predict_instances(self, instances: torch.Tensor) -> torch.Tensor:
# #         """Predict instance-level scores"""
# #         batch_size, num_instances, _ = instances.shape
# #         instances_flat = instances.view(-1, self.input_dim)
# #
# #         # Get instance encodings
# #         encoded_instances = self.instance_encoder(instances_flat)
# #
# #         # Instance-level predictions
# #         instance_logits = self.bag_classifier(encoded_instances)
# #         instance_predictions = self.activation(instance_logits)
# #
# #         return instance_predictions.view(batch_size, num_instances, self.num_classes)
# #
# #
# # class CodeBERTMILIntegrated(nn.Module):
# #     """Integration of CodeBERT with Multi-Instance Learning for hunk-level analysis"""
# #
# #     def __init__(self,
# #                  codebert_model: nn.Module,
# #                  mil_input_dim: int = 768,
# #                  mil_hidden_dim: int = 256):
# #         super().__init__()
# #
# #         self.codebert = codebert_model
# #         self.mil_model = MultiInstanceLearning(
# #             input_dim=mil_input_dim,
# #             hidden_dim=mil_hidden_dim
# #         )
# #
# #         # Feature projection layer
# #         self.feature_projection = nn.Linear(
# #             self.codebert.hidden_size,
# #             mil_input_dim
# #         )
# #
# #     def forward(self,
# #                 batch_input_ids: List[torch.Tensor],
# #                 batch_attention_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
# #         """
# #         Forward pass for CodeBERT-MIL integration
# #
# #         Args:
# #             batch_input_ids: List of tensors, each [num_hunks, seq_len]
# #             batch_attention_masks: List of tensors, each [num_hunks, seq_len]
# #         """
# #         batch_size = len(batch_input_ids)
# #         batch_features = []
# #
# #         # Process each commit's hunks through CodeBERT
# #         for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
# #             if input_ids.size(0) == 0:  # No hunks
# #                 # Create dummy features
# #                 hunk_features = torch.zeros(1, self.codebert.hidden_size)
# #             else:
# #                 # Get CodeBERT features for all hunks
# #                 with torch.no_grad():
# #                     outputs = self.codebert.codebert(
# #                         input_ids=input_ids,
# #                         attention_mask=attention_mask,
# #                         return_dict=True
# #                     )
# #                     hunk_features = outputs.pooler_output  # [num_hunks, hidden_size]
# #
# #             # Project features to MIL input dimension
# #             projected_features = self.feature_projection(hunk_features)
# #             batch_features.append(projected_features)
# #
# #         # Pad sequences to same length
# #         max_hunks = max([f.size(0) for f in batch_features])
# #         padded_features = []
# #
# #         for features in batch_features:
# #             num_hunks = features.size(0)
# #             if num_hunks < max_hunks:
# #                 padding = torch.zeros(max_hunks - num_hunks, features.size(1))
# #                 features = torch.cat([features, padding], dim=0)
# #             padded_features.append(features)
# #
# #         # Stack into tensor [batch_size, max_hunks, feature_dim]
# #         instance_features = torch.stack(padded_features, dim=0)
# #
# #         # Apply MIL
# #         bag_predictions, attention_weights = self.mil_model(instance_features)
# #
# #         return {
# #             'bag_predictions': bag_predictions,
# #             'attention_weights': attention_weights,
# #             'instance_features': instance_features
# #         }
#
#
#
#
#
#
# import pytest
# import torch
# import numpy as np
# from src.models.codebert_model import CodeBERTMultiTask
# from src.models.baselines import BaselineModels
# from src.models.mil_model import MultiInstanceLearning
#
#
# class TestCodeBERTModel:
#     def test_model_initialization(self):
#         model = CodeBERTMultiTask()
#         assert model is not None
#         assert hasattr(model, 'codebert')
#         assert hasattr(model, 'quality_predictor')
#
#     def test_forward_pass(self):
#         model = CodeBERTMultiTask()
#         input_ids = torch.randint(0, 1000, (2, 128))
#         attention_mask = torch.ones(2, 128)
#
#         outputs = model(input_ids, attention_mask)
#         assert 'quality_score' in outputs
#         assert outputs['quality_score'].shape == (2, 1)
#
#     def test_single_task_forward(self):
#         model = CodeBERTMultiTask()
#         input_ids = torch.randint(0, 1000, (1, 128))
#         attention_mask = torch.ones(1, 128)
#
#         outputs = model(input_ids, attention_mask, task='quality')
#         assert 'quality_score' in outputs
#         assert 'comment_logits' not in outputs
#
#
# class TestBaselineModels:
#     def test_feature_preparation(self):
#         baseline = BaselineModels()
#         feature_dicts = [
#             {'feature1': 1, 'feature2': 2},
#             {'feature1': 3, 'feature2': 4}
#         ]
#
#         X = baseline.prepare_features(feature_dicts)
#         assert X.shape == (2, 2)
#
#     def test_model_training(self):
#         baseline = BaselineModels()
#         X = np.random.rand(100, 10)
#         y = np.random.randint(0, 2, 100)
#
#         models = baseline.train_all_models(X, y)
#         assert 'logistic' in models
#         assert 'random_forest' in models
#         assert 'svm' in models
#
#
# class TestMILModel:
#     def test_mil_initialization(self):
#         model = MultiInstanceLearning(input_dim=100)
#         assert model.input_dim == 100
#         assert model.hidden_dim == 256
#
#     def test_mil_forward(self):
#         model = MultiInstanceLearning(input_dim=50)
#         instances = torch.randn(2, 10, 50)  # batch_size=2, num_instances=10
#
#         predictions, attention = model(instances)
#         assert predictions.shape == (2, 1)
#         assert attention.shape == (2, 10, 1)


import os
import sys
import json
import base64
import hashlib
import requests
from datetime import datetime


class BackblazeB2:
    def __init__(self):
        self.application_key_id = os.getenv('B2_APPLICATION_KEY_ID')
        self.application_key = os.getenv('B2_APPLICATION_KEY')
        self.bucket_name = os.getenv('B2_BUCKET_NAME')
        self.api_url = 'https://api.backblazeb2.com'
        self.auth_token = None
        self.upload_url = None
        self.upload_auth_token = None

    # In the authenticate method, add debug prints:
    def authenticate(self):
        """Authenticate with Backblaze B2 API"""
        if not all([self.application_key_id, self.application_key]):
            print('❌ Backblaze B2 credentials not configured!')
            print('Required: B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME')
            return False

        try:
            # Create basic auth header
            credentials = f"{self.application_key_id}:{self.application_key}"
            print(f"Debug - credentials: {credentials[:20]}...")  # debug
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            print(f"Debug - encoded: {encoded_credentials[:20]}...")  # debug

            headers = {
                'Authorization': f'Basic {encoded_credentials}'
            }

            print(f"Debug - making request to: {self.api_url}/b2api/v2/b2_authorize_account")
            response = requests.get(f'{self.api_url}/b2api/v2/b2_authorize_account', headers=headers)

            print(f"Debug - response status: {response.status_code}")
            print(f"Debug - response text: {response.text}")

            if response.status_code == 200:
                data = response.json()
                self.api_url = data['apiUrl']
                self.auth_token = data['authorizationToken']
                print("✅ Authentication successful")
                return True
            else:
                print(f"❌ B2 authentication failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except Exception as e:
            print(f'❌ Authentication error: {str(e)}')
            return False

    def get_upload_url(self):
        """Get upload URL for the bucket"""
        try:
            # First, get bucket ID
            headers = {
                'Authorization': self.auth_token
            }

            data = {
                'accountId': self.application_key_id[:12],
                'bucketName': self.bucket_name
            }

            response = requests.post(
                f'{self.api_url}/b2api/v2/b2_list_buckets',
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                print(f'❌ Failed to list buckets: {response.status_code}')
                print(f'Response: {response.text}')
                return False

            buckets = response.json().get('buckets', [])
            bucket = next((b for b in buckets if b['bucketName'] == self.bucket_name), None)

            if not bucket:
                print(f'❌ Bucket "{self.bucket_name}" not found!')
                print('Available buckets:', [b['bucketName'] for b in buckets])
                return False

            bucket_id = bucket['bucketId']

            # Get upload URL
            data = {'bucketId': bucket_id}
            response = requests.post(
                f'{self.api_url}/b2api/v2/b2_get_upload_url',
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                upload_data = response.json()
                self.upload_url = upload_data['uploadUrl']
                self.upload_auth_token = upload_data['authorizationToken']
                print('✅ Upload URL obtained')
                return True
            else:
                print(f'❌ Failed to get upload URL: {response.status_code}')
                print(f'Response: {response.text}')
                return False

        except Exception as e:
            print(f'❌ Error getting upload URL: {str(e)}')
            return False

    def calculate_sha1(self, file_path):
        """Calculate SHA1 hash of file"""
        sha1_hash = hashlib.sha1()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha1_hash.update(chunk)
        return sha1_hash.hexdigest()

    def upload_file(self, file_path):
        """Upload file to Backblaze B2"""
        try:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            print(f"🔄 Uploading {file_name} to B2...")
            print(f"📊 File size: {file_size / (1024 * 1024):.2f} MB")

            # Calculate SHA1 hash
            print("🔐 Calculating file hash...")
            sha1_hash = self.calculate_sha1(file_path)

            # Prepare headers
            headers = {
                'Authorization': self.upload_auth_token,
                'X-Bz-File-Name': file_name,
                'Content-Type': 'application/gzip',
                'Content-Length': str(file_size),
                'X-Bz-Content-Sha1': sha1_hash,
                'X-Bz-Info-src_last_modified_millis': str(int(os.path.getmtime(file_path) * 1000)),
                'X-Bz-Info-backup_type': 'database',
                'X-Bz-Info-server': os.getenv('HOSTNAME', 'unknown'),
                'X-Bz-Info-timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }

            # Upload file
            with open(file_path, 'rb') as file_data:
                response = requests.post(
                    self.upload_url,
                    headers=headers,
                    data=file_data
                )

            if response.status_code == 200:
                upload_result = response.json()
                print(f'✅ Upload successful!')
                print(f'📁 File ID: {upload_result["fileId"]}')
                print(f'🔗 File Name: {upload_result["fileName"]}')
                print(f'📊 Uploaded Size: {upload_result["contentLength"]} bytes')
                return True
            else:
                print(f'❌ Upload failed: {response.status_code}')
                print(f'Response: {response.text}')
                return False

        except Exception as e:
            print(f'❌ Upload error: {str(e)}')
            return False

    def cleanup_old_backups(self, retention_days=30):
        """Delete backups older than retention_days"""
        try:
            headers = {
                'Authorization': self.auth_token
            }

            # List files in bucket
            data = {
                'bucketName': self.bucket_name,
                'maxFileCount': 1000
            }

            response = requests.post(
                f'{self.api_url}/b2api/v2/b2_list_file_names',
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                print(f'⚠️ Could not list files for cleanup: {response.status_code}')
                return False

            files = response.json().get('files', [])
            current_time = datetime.now().timestamp() * 1000  # B2 uses milliseconds
            retention_ms = retention_days * 24 * 60 * 60 * 1000

            old_files = []
            for file_info in files:
                file_time = int(file_info.get('fileInfo', {}).get('src_last_modified_millis', 0))
                if current_time - file_time > retention_ms:
                    old_files.append(file_info)

            if old_files:
                print(f'🧹 Found {len(old_files)} old backup files to delete')
                deleted_count = 0

                for file_info in old_files:
                    delete_data = {
                        'fileId': file_info['fileId'],
                        'fileName': file_info['fileName']
                    }

                    delete_response = requests.post(
                        f'{self.api_url}/b2api/v2/b2_delete_file_version',
                        headers=headers,
                        json=delete_data
                    )

                    if delete_response.status_code == 200:
                        deleted_count += 1
                        print(f'🗑️ Deleted: {file_info["fileName"]}')
                    else:
                        print(f'⚠️ Failed to delete: {file_info["fileName"]}')

                print(f'✅ Cleanup completed: {deleted_count}/{len(old_files)} files deleted')
            else:
                print('✅ No old files to cleanup')

            return True

        except Exception as e:
            print(f'⚠️ Cleanup error: {str(e)}')
            return False


def upload_to_b2(file_path):
    """Main upload function"""
    if not os.path.exists(file_path):
        print(f'❌ File not found: {file_path}')
        return False
    b2 = BackblazeB2()
    # Authenticate
    if not b2.authenticate():
        return False
    # Get upload URL
    if not b2.get_upload_url():
        return False
    # Upload file
    if not b2.upload_file(file_path):
        return False
    # Cleanup old backups
    retention_days = int(os.getenv('BACKUP_RETENTION_DAYS', '30'))
    b2.cleanup_old_backups(retention_days)
    return True


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 b2_upload.py <file_path>')
        sys.exit(1)
    file_path = sys.argv[1]
    if upload_to_b2(file_path):
        print('✅ B2 upload completed successfully')
    else:
        print('❌ B2 upload failed')
        sys.exit(1)