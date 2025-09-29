#!/bin/bash
# Activation script for NoCap-Test environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nocap-test
echo "NoCap-Test environment activated!"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
