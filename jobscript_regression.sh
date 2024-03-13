#!/bin/sh
#BSUB -J Regression
#BSUB -o Regression%J.out
#BSUB -e Regression%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8G]"
#BSUB -W 4:00
#BSUB -N
# end of BSUB options

module load python3/3.11.7

# load CUDA (for GPU support)
module load cuda/11.8

# activate the virtual environment
source MT/bin/activate

python src/train_regression.py experiments=train_regression_mimbo
