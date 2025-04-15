#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 56
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=m25031

# Changer le répertoire de travail
cd ../hover_net

# Lancer l’entraînement
# L'argument --config est un exemple, adaptez-le en fonction de la manière dont vous configurez vos paramètres d'entraînement
uv run ./run_train.py --gpu=0,1,2,3