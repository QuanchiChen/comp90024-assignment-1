#!/bin/bash

#SBATCH --partition=physical
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

module load mpi4py/3.0.2-timed-pingpong
module load python/3.7.4

source ~/venvs/venv-3.7.4/bin/activate
mpirun python main.py ~/bigTwitter.json ~/sal.json
