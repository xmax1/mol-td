#!/bin/bash -ex
#SBATCH --partition=sm3090
#SBATCH -N 1-1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00 # 2 days of runtime (can be set to 7 days)
#SBATCH -o ./slurm/output.%j.out # STDOUT

source ~/.bashrc

conda activate td

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/energy/amawi/projects/mol-td

python train.py "$@"
