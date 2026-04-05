#!/bin/bash
#SBATCH --job-name=and
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err
#SBATCH --gpus=1
#SBATCH --qos=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liys@a-star.edu.sg

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Activate virtual environment
source .env/bin/activate

# Apply the requested environment variables
export HOME=/homes/yusha
export MPLCONFIGDIR=/homes/yusha/.config/matplotlib
export TRANSFORMERS_CACHE=/homes/yusha/.cache/huggingface

# Example command:
python run_anomalydino.py \
    --image_path "path/to/your/image.jpg" \
    --prompt "describe the defect" \
    --save_path "./results_custom"

echo "Job finished at $(date)"
