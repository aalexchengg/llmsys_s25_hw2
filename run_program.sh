#!/bin/bash
#SBATCH --job-name=problem4
#SBATCH --output=out/problem4.out
#SBATCH --error=out/problem4.err
#SBATCH --partition=general
#SBATCH --time=1-12:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1

# Your job commands go here

# Load in cuda
source /etc/profile.d/modules.sh
module load cuda-12.4
nvcc --version
source ~/.bashrc

# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate minitorch
# Install pycuda
pip install pycuda

# compile_cuda.sh
mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
# verify installation
python3 -m install
# Problem 1
# python3 -m pytest -l -v -k "test_logsumexp_student"
# python3 -m pytest -l -v -k "test_softmax_loss_student"
# Problem 2
# python3 -m pytest -l -v -k "test_linear_student"
# python3 -m pytest -l -v -k "test_dropout_student"
# python3 -m pytest -l -v -k "test_layernorm_student"
# python3 -m pytest -l -v -k "test_embedding_student"
# Problem 3
# python3 -m pytest -l -v -k "test_multihead_attention_student"
# python3 -m pytest -l -v -k "test_transformer_layer_1_student"
# python3 -m pytest -l -v -k "test_transformer_layer_2_student"
# python3 -m pytest -l -v -k "test_decoder_lm_student"
# Problem 4
python3 project/run_machine_translation.py
