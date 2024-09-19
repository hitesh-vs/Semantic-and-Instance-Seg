#!/bin/bash

SBATCH --mail-user=sviswasam@wpi.edu
SBATCH --mail-type=ALL

SBATCH -J window_seg
SBATCH --output=/home/sviswasam/p2/project2/logs/window_seg%j.out
SBATCH --error=/home/sviswasam/p2/project2/logs/window_seg%j.err

SBATCH -N 1
SBATCH -n 8
SBATCH --mem=64G
SBATCH --gres=gpu:1
SBATCH -C H100|A30|V100
SBATCH -A rbe595
SBATCH -p academic
SBATCH -t 23:00:00

python3 train.py
