#!/bin/bash -ex
#SBATCH --partition=xeon8
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=0-10:00:00 # 2 days of runtime (can be set to 7 days)
#SBATCH -o ./slurm/output.%j.out # STDOUT

dataset=$1

source ~/.bashrc

module load GPAW

conda activate td

cd /home/energy/amawi/projects/mol-td

python mol_td/compute_energy.py -d $dataset
