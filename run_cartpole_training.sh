#!/bin/bash

# Generate timestamp
timestamp=$(date +"%Y%m%d-%H%M%S")
log_dir="logs/dqn_${timestamp}"

# Create the directory
mkdir -p "$log_dir"

# activate the python environment
source ~/rl-env/bin/activate

# Run the script with the log directory as an argument and capture output
python -u cartpole.py --log-dir="$log_dir" --model="saved_models/initial_model.keras" 2>&1 | tee "${log_dir}/run_transcript.log"

# -u argument means "unbuffered" so output goes straight to the terminal