#!/bin/bash
#SBATCH --partition=ga80-1gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j

module load nvhpc

mpiexec -n 1 ./program

