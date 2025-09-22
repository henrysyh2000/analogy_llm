#!/bin/bash -l

# slurm submit script
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Northstar
# SBATCH --partition=Sasquatch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=50:00:00
#SBATCH --export=ALL

# Pick the GPU that is free
export CUDA_VISIBLE_DEVICES=4

# for northstar cluster
source ~/dev_env/bin/activate
echo "dev_env activated!"
python3 -V

# for sasquatch cluster, need to update python to 3.12. Remember next time create virtual env always inside a computing node. Login node has different python version.
# source ~/sasquatch_venv/bin/activate
# echo "sasquatch_venv activated!"
# python3 -V


# test inference code
python3 inference.py --model "openai/gpt-oss-20b" --verbose True --batch_size 5