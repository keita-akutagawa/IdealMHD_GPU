#!/bin/bash
#SBATCH --partition=ga40-2gpu
#SBATCH --gres=gpu:2
#SBATCH --time=0:10:00
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j

module load nvhpc

mpiexec -n 2 ./program

