#!/bin/bash

#SBATCH --job-name=cellvit_train    # Nom du job
#SBATCH --output=cellvit_train_%j.out # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=cellvit_train_%j.err  # Fichier de sortie d'erreur (%j = ID du job)
#SBATCH -p mesonet                  # Nom de la partition (obligatoire)
#SBATCH --account=m25031            # !! REMPLACEZ m2xxxx par votre ID de projet !! (obligatoire)
#SBATCH --nodes=1                   # Nombre de nœuds requis (1 est suffisant pour ce script)
#SBATCH --ntasks-per-node=1         # Nombre de tâches (processus) par nœud (1 script python)
#SBATCH --cpus-per-task=8           # Nombre de cœurs CPU par tâche (ajustable)
#SBATCH --gres=gpu:1                # Nombre et type de GPUs requis (1 GPU générique)
#SBATCH --mem=64G                   # Mémoire RAM requise par nœud (ex: 64 Go, ajustable)
#SBATCH --time=10:00:00             # Temps maximum d'exécution (HH:MM:SS, max 10 jours)

# ----- Configuration de l'environnement -----
echo "Date: $(date)"
echo "Noeud: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Compte: $SLURM_JOB_ACCOUNT"
echo "GPUs alloués: $CUDA_VISIBLE_DEVICES"

# Charger les modules nécessaires (À VÉRIFIER/ADAPTER selon la configuration de Juliet)
# module purge # Commencer avec un environnement propre
# module load cuda/XXX # Charger la version CUDA appropriée si nécessaire
# module load cudnn/YYY # Charger cuDNN si nécessaire
# module load anaconda/ZZZ # Charger Anaconda/Miniconda si nécessaire

# Activer l'environnement virtuel uv (si vous en utilisez un)
# Assurez-vous que le répertoire .venv existe et contient l'environnement
VENV_PATH="/home/queberal/Deep_learning_project/cellvit_plus_plus/.venv" # Ajustez si nécessaire
if [ -d "${VENV_PATH}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "Activation de l'environnement uv: ${VENV_PATH}"
    source ${VENV_PATH}/bin/activate
    
    # Vérifier la version CUDA disponible
    echo "Vérification de la version CUDA avec nvidia-smi..."
    nvidia-smi
    
    # Installer cupy (adapter la version cuda si nécessaire, ex: cupy-cuda11x, cupy-cuda12x)
    echo "Installation de cupy pour CUDA 12.x..."
    uv pip install cupy-cuda12x
    if [ $? -ne 0 ]; then
        echo "Erreur lors de l'installation de cupy. Vérifiez la version CUDA du cluster."
        exit 1
    fi
else
    echo "Erreur: Environnement uv non trouvé à ${VENV_PATH}"
    exit 1
fi

# ----- Exécution du script d'entraînement -----
echo "Lancement de l'entraînement CellViT++..."

# Chemin absolu vers le script et le fichier de configuration
# !! ASSUREZ-VOUS QUE CES CHEMINS SONT CORRECTS SUR JULIET !!
SCRIPT_PATH="/home/queberal/Deep_learning_project/cellvit_plus_plus/CellViT-plus-plus/cellvit/train_cell_classifier_head.py"
CONFIG_PATH="/home/queberal/Deep_learning_project/cellvit_plus_plus/training-configuration.yaml"

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
echo "Fin du script Slurm."
echo "Date: $(date)"

exit ${EXIT_CODE}
