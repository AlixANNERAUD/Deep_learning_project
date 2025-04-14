#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 56
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=m25031

# Optionnel : Chargement de modules
module load anaconda3                      # Chargement du module Anaconda, si votre cluster fonctionne ainsi

# Activation de l’environnement Conda dédié à Hovernet
source activate hovernet_env

# Lancer l’entraînement
# L'argument --config est un exemple, adaptez-le en fonction de la manière dont vous configurez vos paramètres d'entraînement
python run_train.py \
    --gpu=0,1,2,3 \
    --nr_types=todo \
    --type_info_path=todo \
    --model_path=../weight/
    --model_mode=fast \
    --nr_inference_workers=28\
    --nr_post_proc_workers=56 \
    --batch_size=128 \
    --dataset /chemin/vers/votre/dataset
