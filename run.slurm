#!/bin/bash
#SBATCH -n1
#SBATCH -c1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --mem=100000

python -u tf-recon.py
