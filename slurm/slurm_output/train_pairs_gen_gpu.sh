#!/bin/sh
#SBATCH --job-name=train_pair_gen_gpu # create a short name for your job
#SBATCH --output=slurm/slurm_output/train_pair_gen_gpu/%x_%j.out
#SBATCH --error=slurm/slurm_output/train_pair_gen_gpu/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN,END,FAIL          # send email on job start, end and on fail
#SBATCH --mail-user=ds6237@princeton.edu #--> ADD YOUR EMAIL

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

module purge
module load anaconda3/2023.9
module load cudatoolkit/12.4
module load gcc-toolset/10

conda activate torch-env

log_info "Python version: $(python --version 2>&1)"

python -c "import torch; print(f'PyTorch version: {torch.version.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Print the GPU information
if command -v nvidia-smi &>/dev/null; then
    log_info "GPU Information: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
else
    log_info "CUDA not installed or GPUs not available."
fi

# Run the training script 10000 times
for i in {1..10000}
do
    log_info "Running iteration $i of 10000"
    python train_gen_stu.py
done