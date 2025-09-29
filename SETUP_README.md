# NoCap-Test Environment Setup

This document explains how to set up the conda environment for the NoCap-Test project.

## Prerequisites

1. **Conda/Miniconda**: Make sure you have conda installed on your system
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Or install Anaconda: https://www.anaconda.com/products/distribution

2. **CUDA (Optional but recommended)**: For GPU acceleration
   - Install CUDA drivers and toolkit
   - The script will install PyTorch with CUDA support

## Quick Setup

1. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

2. **Activate the environment**:
   ```bash
   source activate_env.sh
   # or
   conda activate nocap-test
   ```

3. **Run the training**:
   ```bash
   ./run.sh
   ```

## What the Setup Script Does

- Creates a conda environment named `nocap-test` with Python 3.11
- Installs PyTorch with CUDA 12.1 support
- Installs essential ML packages (numpy, pandas, scipy, etc.)
- Installs project-specific dependencies (wandb, datasets, etc.)
- Installs development tools (black, flake8, pytest)
- Installs additional dependencies from `requirements.txt` if present
- Creates an activation script for easy environment switching

## Manual Setup (Alternative)

If you prefer to set up the environment manually:

```bash
# Create environment
conda create -n nocap-test python=3.11 -y

# Activate environment
conda activate nocap-test

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other packages
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn jupyter -y
pip install wandb datasets huggingface-hub tqdm pyyaml click

# Install from requirements.txt
pip install -r requirements.txt
```

## Troubleshooting

### CUDA Issues
- Make sure you have CUDA-compatible GPU drivers installed
- Check CUDA version compatibility with PyTorch
- If CUDA is not available, PyTorch will fall back to CPU-only mode

### Environment Issues
- If the environment already exists, the script will remove and recreate it
- Make sure you have sufficient disk space for the environment (~2-3 GB)

### Permission Issues
- Make sure the setup script is executable: `chmod +x setup.sh`
- Run with appropriate permissions if needed

## Verification

After setup, verify the installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

This should show your PyTorch version and whether CUDA is available.
