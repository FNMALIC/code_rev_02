# AI-Powered Automated Code Review System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An intelligent automated code review system that combines multiple AI approaches to analyze code quality, detect issues, and provide actionable feedback. The system integrates seamlessly with Git workflows and uses advanced machine learning models including CodeBERT for deep code understanding.

## 🚀 Features

### Core Capabilities
- **Multi-Modal Analysis**: Combines AST parsing, graph analysis, and transformer-based semantic understanding
- **Git Integration**: Analyzes commits, pull requests, and code diffs
- **Multi-Task Learning**: Single model handles quality assessment, comment generation, and code refinement
- **Security Analysis**: Detects potential security vulnerabilities and code smells
- **Interpretable AI**: SHAP explanations and attention visualizations for model decisions

### Analysis Types
- **Structural Analysis**: AST-based feature extraction (complexity, patterns, code smells)
- **Semantic Understanding**: CodeBERT-powered deep code comprehension
- **Graph Analysis**: Control flow and dependency relationship modeling
- **Quality Scoring**: Automated code quality assessment with detailed feedback

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Git Diff      │───▶│  Feature         │───▶│   Multi-Task    │
│   Parser        │    │  Extraction      │    │   CodeBERT      │
└─────────────────┘    │                  │    │   Model         │
                       │ • AST Features   │    └─────────────────┘
┌─────────────────┐    │ • Graph Features │           │
│   Code Input    │───▶│ • CodeBERT       │           ▼
└─────────────────┘    │   Embeddings     │    ┌─────────────────┐
                       └──────────────────┘    │   Outputs       │
                                              │ • Quality Score │
                                              │ • Comments      │
                                              │ • Refinements   │
                                              └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/FNMALIC/code_rev_02.git
cd code_rev_02
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Download required models**
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/codebert-base')"
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# External AI API Configuration
EXTERNAL_AI_API_KEY=your_api_key_here
EXTERNAL_AI_MODEL=gpt-3.5-turbo

# Model Configuration
CODEBERT_MODEL=microsoft/codebert-base
MAX_SEQUENCE_LENGTH=512

# Training Configuration
BATCH_SIZE=8
LEARNING_RATE=2e-5
NUM_EPOCHS=5

# Data Configuration
DATA_LIMIT=500
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1
TEST_SPLIT=0.1
```

### Configuration File
Modify `configs/config.yaml` for detailed settings:

```yaml
model:
  name: "microsoft/codebert-base"
  hidden_dim: 768
  dropout: 0.1

training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 5
  weight_decay: 0.01

evaluation:
  metrics: ["accuracy", "f1", "precision", "recall", "auc"]
  generate_explanations: true
```

## 🚀 Usage

### Command Line Interface

**Analyze a single file:**
```bash
python main.py --mode analyze --file path/to/your/code.py
```

**Analyze Git repository:**
```bash
python main.py --mode git --repo /path/to/repo --branch main
```

**Train the model:**
```bash
python main.py --mode train --data /path/to/training/data
```

**Evaluate model performance:**
```bash
python main.py --mode evaluate --model /path/to/model --test-data /path/to/test/data
```

### Python API

```python
from src.models.codebert_model import CodeBERTMultiTask
from src.features.ast_features import ASTFeatureExtractor
from src.integration.git_integration import GitIntegration

# Initialize the model
model = CodeBERTMultiTask()

# Analyze code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Extract features
ast_extractor = ASTFeatureExtractor()
features = ast_extractor.extract_features(code)

# Get quality assessment
quality_score = model.predict_quality(code)
print(f"Code quality score: {quality_score}")
```

### Git Integration

```python
from src.integration.git_integration import GitIntegration

# Analyze repository
git_integration = GitIntegration("/path/to/repo")
commits = git_integration.get_recent_commits(limit=10)

for commit in commits:
    analysis = git_integration.analyze_commit(commit.hexsha)
    print(f"Commit {commit.hexsha}: Quality Score {analysis['quality_score']}")
```

## 📁 Project Structure

```
code_rev_02/
├── src/
│   ├── data/                 # Data loading and preprocessing
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/             # Feature extraction modules
│   │   ├── ast_features.py
│   │   ├── graph_features.py
│   │   └── enhanced_labeling.py
│   ├── models/               # ML models and architectures
│   │   ├── codebert_model.py
│   │   ├── baselines.py
│   │   └── mil_model.py
│   ├── evaluation/           # Model evaluation and metrics
│   │   ├── metrics.py
│   │   └── interpretability.py
│   ├── integration/          # Git and external integrations
│   │   ├── git_integration.py
│   │   └── diff_parser.py
│   └── utils/                # Utility functions
│       ├── logging_config.py
│       └── ExternalAIReviewer.py
├── configs/                  # Configuration files
│   └── config.yaml
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Add type hints for function parameters and return values
- Write docstrings for all public functions and classes

## 📈 Roadmap

- [ ] **Web Interface**: React-based dashboard for code review results
- [ ] **IDE Plugins**: VSCode and PyCharm extensions
- [ ] **CI/CD Integration**: GitHub Actions and Jenkins plugins
- [ ] **Multi-Language Support**: JavaScript, Java, C++ analysis
- [ ] **Advanced Security**: SAST integration and vulnerability database
- [ ] **Performance Optimization**: Model quantization and caching
- [ ] **Collaborative Features**: Team review workflows and approval systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft CodeBERT**: Pre-trained model for code understanding
- **Hugging Face Transformers**: Transformer model implementations
- **scikit-learn**: Machine learning utilities and baseline models
- **NetworkX**: Graph analysis capabilities
- **SHAP**: Model interpretability and explanations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/FNMALIC/code_rev_02/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FNMALIC/code_rev_02/discussions)
- **Email**: fonkou.nixon@example.com

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{automated_code_review_2024,
  title={AI-Powered Automated Code Review System},
  author={Fonkou Nixon},
  year={2024},
  url={https://github.com/FNMALIC/code_rev_02}
}
```

---

**Made with ❤️ by Fonkou Nixon**
