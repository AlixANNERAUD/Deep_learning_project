#!/bin/bash

#SBATCH --job-name=cellvit_train    # Nom du job
#SBATCH --output=cellvit_train_%j.out # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=cellvit_train_%j.err  # Fichier de sortie d'erreur (%j = ID du job)
#SBATCH -p mesonet                  # Nom de la partition (obligatoire)
#SBATCH --account=m25031            # !! REMPLACEZ m2xxxx par votre ID de projet !! (obligatoire)
#SBATCH --nodes=1                   # Nombre de nœuds requis (1 est suffisant pour ce script)
#SBATCH --ntasks-per-node=1         # Nombre de tâches (processus) par nœud (1 script python)
#SBATCH --cpus-per-task=8           # Nombre de cœurs CPU par tâche (ajustable)
#SBATCH --gres=gpu:a100:1           # Nombre et type de GPUs requis (1 A100)
#SBATCH --mem=64G                   # Mémoire RAM requise par nœud (ex: 64 Go, ajustable)
#SBATCH --time=12:00:00             # Temps maximum d'exécution (HH:MM:SS, max 10 jours)

# ----- Configuration de l'environnement -----
echo "Date: $(date)"
echo "Noeud: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Compte: $SLURM_JOB_ACCOUNT"
echo "GPUs alloués: $CUDA_VISIBLE_DEVICES"

# Charger les modules nécessaires (À VÉRIFIER/ADAPTER selon la configuration de Juliet)
module purge # Commencer avec un environnement propre
# module load cuda/XXX # Charger la version CUDA appropriée si nécessaire
# module load cudnn/YYY # Charger cuDNN si nécessaire
# module load anaconda/ZZZ # Charger Anaconda/Miniconda si nécessaire

# Activer l'environnement Conda (si vous en utilisez un)
# Assurez-vous que 'conda' est dans votre PATH ou chargez le module approprié
# Le chemin peut varier, vérifiez avec 'which conda' ou 'conda info --base'
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate cellvit_env # Remplacer par le nom exact de votre environnement

# ----- Exécution du script d'entraînement -----
echo "Lancement de l'entraînement CellViT++..."

# Chemin absolu vers le script et le fichier de configuration
# !! ASSUREZ-VOUS QUE CES CHEMINS SONT CORRECTS SUR JULIET !!
SCRIPT_PATH="/chemin/absolu/vers/cellvit_plus_plus/CellViT-plus-plus/cellvit/train_cell_classifier_head.py"
CONFIG_PATH="/chemin/absolu/vers/votre/training_config.yaml" # <-- !! REMPLACEZ CECI !!

# Vérifier si les fichiers existent
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "Erreur: Le script Python ${SCRIPT_PATH} n'a pas été trouvé."
    exit 1
fi
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Erreur: Le fichier de configuration ${CONFIG_PATH} n'a pas été trouvé."
    exit 1
fi

# Lancer l'entraînement
python3 ${SCRIPT_PATH} --config ${CONFIG_PATH}

# Capturer le code de sortie
EXIT_CODE=$?
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Erreur: L'entraînement Python s'est terminé avec le code ${EXIT_CODE}"
else
    echo "Entraînement terminé avec succès."
fi

# ----- Fin -----
conda deactivate
echo "Fin du script Slurm."
echo "Date: $(date)"

exit ${EXIT_CODE}
