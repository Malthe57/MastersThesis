#!/bin/sh
#BSUB -J C_inference
#BSUB -o C_inference%J.out
#BSUB -e C_inference%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -W 4:00
#BSUB -N
# end of BSUB options

module load python3/3.11.7

# load CUDA (for GPU support)
module load cuda/11.8

# activate the virtual environment
source MT/bin/activate

python src/inference_classification.py --model_name "C_MIMO" --Ms "1,2" --n_classes "100" --reps "5" --resnet
