#!/bin/bash -ex
#SBATCH --partition=sm3090
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --time=0-01:00:00 # 2 days of runtime (can be set to 7 days)
#SBATCH -o ./slurm/output.%j.out # STDOUT
# #SBATCH --gres=gpu:RTX3090

source ~/.bashrc

# module load foss
# # #SBATCH -o junk.out
# nvidia-smi
# nvcc --version
conda activate td

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/energy/amawi/projects/mol-td

python train.py "$@"
