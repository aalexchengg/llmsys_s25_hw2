#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --output=out/task2.out
#SBATCH --error=out/task2.err
#SBATCH --partition=general
#SBATCH --time=5:00:00
#SBATCH --gpus=1

# Your job commands go here
source /etc/profile.d/modules.sh
module load cuda-12.4
nvcc --version
source activate
conda activate minitorch
pip install pycuda

# compile_cuda.sh
mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
# verify installation
python3 -m install
# run script
python3 -m pytest -l -v -k "test_logsumexp_student"
python -m pytest -l -v -k "test_softmax_loss_student"