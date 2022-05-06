#!/bin/bash -ex
#SBATCH --partition=xeon16
#SBATCH -N 1
#SBATCH --time=0-10:00:00 # 2 days of runtime (can be set to 7 days)
#SBATCH -o ./slurm/output.%j.out # STDOUT

dataset=$1

source ~/.bashrc

module purge 

module load GPAW

conda activate td

cd /home/energy/amawi/projects/mol-td

python compute_energy.py -d $dataset
