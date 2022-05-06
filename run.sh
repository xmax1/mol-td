#!/bin/bash -ex
#SBATCH --partition=sm3090
#SBATCH -N 1-1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00 # 2 days of runtime (can be set to 7 days)
#SBATCH -o ./slurm/output.%j.out # STDOUT

arr=()
while IFS= read -r line; do arr+=("$line"); done < experiments.txt
cmd=${arr[$SLURM_ARRAY_TASK_ID]}
len=$((${#arr[@]}-1))
echo $cmd

source ~/.bashrc

module purge 

module load GCC
module load CUDA/11.4.1
module load cuDNN

conda activate td

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/energy/amawi/projects/mol-td

python main.py $cmd
