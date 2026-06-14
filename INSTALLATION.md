# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Farazahmedsyed/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Create Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import xgboost, shap, streamlit; print('All packages installed successfully!')"
```

## Running the System

### Start Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

### Run Tests

```bash
pytest tests/ -v
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Ensure virtual environment is activated
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Issue: Port 8501 already in use

**Solution**: Specify different port
```bash
streamlit run dashboard/app.py --server.port=8502
```

### Issue: XGBoost installation fails

**Solution**: Install from conda
```bash
conda install -c conda-forge xgboost
```

## Optional Dependencies

### For Jupyter Notebooks

```bash
pip install jupyter notebook
jupyter notebook
```

### For Development

```bash
pip install pytest pytest-cov black flake8
```

## Docker Installation

### Build Docker Image

```bash
docker build -t fraud-detection-system .
```

### Run Container

```bash
docker run -p 8501:8501 fraud-detection-system
```

## Next Steps

1. See [README.md](README.md) for usage examples
2. Check [USAGE.md](USAGE.md) for detailed tutorials
3. Review [config/config.yaml](config/config.yaml) for configuration
