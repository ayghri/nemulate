#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Add the project root to PYTHONPATH so nemulate can be imported
export PYTHONPATH="$PWD/..:$PYTHONPATH"

# Default path
CESM_PATH="/buckets/datasets/ssh/simulations/cesm2/monthly/"

# If an argument is provided, use it as the path
if [ -n "$1" ]; then
    CESM_PATH="$1"
fi

# Run the python script
python benchmark_dataloader.py --cesm-path "$CESM_PATH"
