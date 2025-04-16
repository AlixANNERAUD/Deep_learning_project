#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 56
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=m25031

# Changer le répertoire de travail
source ../hover_net/.venv/bin/activate

# Fix error
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Lancer l’entraînement
# L'argument --config est un exemple, adaptez-le en fonction de la manière dont vous configurez vos paramètres d'entraînement
python3 ../hover_net/run_train.py --gpu=0,1