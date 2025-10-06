#!/bin/bash
# setup_cluster_env.sh - Script to set up environment on cluster

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate suite3d-gpu

# Install additional packages if needed
pip install tensorboard tqdm

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import h5py; print(f'h5py version: {h5py.__version__}')"

echo "Environment setup complete!"
