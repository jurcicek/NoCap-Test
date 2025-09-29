#!/bin/bash

# NoCap-Test Conda Environment Setup Script
# This script creates and configures a conda environment for the NoCap-Test project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    print_status "Please install Miniconda or Anaconda first:"
    print_status "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="nocap-test"
PYTHON_VERSION="3.11"

print_status "Setting up conda environment: $ENV_NAME"

# Remove existing environment if it exists
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment
print_status "Creating conda environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Update conda in base environment first
print_status "Updating conda in base environment..."
conda update conda -y

# Activate environment
print_status "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Update pip in the new environment
print_status "Updating pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
# Using CUDA 12.1 as it's compatible with the requirements
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other ML and data science packages
print_status "Installing core ML packages..."
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn jupyter -y

# Install additional packages via pip
print_status "Installing additional packages via pip..."
pip install wandb datasets huggingface-hub tqdm pyyaml click

# Install development tools
print_status "Installing development tools..."
pip install black flake8 pytest

# Optionally install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    print_status "Found requirements.txt, installing additional dependencies..."
    pip install -r requirements.txt
else
    print_warning "No requirements.txt found, skipping additional dependency installation"
fi

# Verify installation
print_status "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Create activation script
print_status "Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation script for NoCap-Test environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nocap-test
echo "NoCap-Test environment activated!"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x activate_env.sh

print_success "Environment setup complete!"
print_status "To activate the environment, run:"
print_status "  source activate_env.sh"
print_status "  or"
print_status "  conda activate $ENV_NAME"

print_status "To run the training script:"
print_status "  ./run.sh"

print_warning "Make sure you have CUDA-compatible GPU and drivers installed for optimal performance."
