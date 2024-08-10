#!/bin/sh
#BSUB -J Classification
#BSUB -o Classification%J.out
#BSUB -e Classification%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -W 6:00
#BSUB -N
# end of BSUB options

module load python3/3.11.7

# load CUDA (for GPU support)
module load cuda/11.8

# activate the virtual environment
source MT/bin/activate

python src/train_classification.py experiments=train_classification_mimbo experiments.hyperparameters.n_subnetworks=2 experiments.hyperparameters.sigma1=40.82
experiments.hyperparameters.batch_size=128 experiments.hyperparameters.is_resnet=True experiments.hyperparameters.dropout_rate=0.0 
experiments.hyperparameters.repetitions=1 experiments.hyperparameters.dataset=CIFAR100 experiments.hyperparameters.learning_rate=1e-3
experiments.hyperparameters.batch_repetition=4 experiments.hyperparameters.gamma=0.1
