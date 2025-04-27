#!/bin/bash

#SBATCH --job-name=nnunet_train # Nom du job
#SBATCH --output=nnunet_train_%j.out # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=nnunet_train_%j.err  # Fichier de sortie d'erreur (%j = ID du job)
#SBATCH -p mesonet                  # Nom de la partition (obligatoire)
#SBATCH --account=m25031            # !! REMPLACEZ m2xxxx par votre ID de projet !! (obligatoire)
#SBATCH --nodes=1                   # Nombre de nœuds requis
#SBATCH --ntasks-per-node=1         # Nombre de tâches (processus) par nœud
#SBATCH --cpus-per-task=8           # Nombre de cœurs CPU par tâche (ajustable)
#SBATCH --mem=64G                   # Mémoire RAM requise par nœud (ajustable)
#SBATCH --time=24:00:00             # Temps maximum d'exécution (HH:MM:SS, ajustable pour training)
#SBATCH --gres=gpu:4                # Nombre de GPUs requis (ajustable)
#SBATCH --nodelist=juliet2

# ----- Configuration de l'environnement ----
echo "Date: $(date)"
echo "Noeud: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Compte: $SLURM_JOB_ACCOUNT"
echo "GPUs alloués: $CUDA_VISIBLE_DEVICES"

# Définir les variables d'environnement nnU-Net
export nnUNet_preprocessed="/home/queberal/Deep_learning_project/nnunet/nnUNet_preprocessed"
export nnUNet_raw="/home/queberal/Deep_learning_project/nnunet/nnUNet_raw"
export nnUNet_results="/home/queberal/Deep_learning_project/nnunet/nnUNet_results"
echo "nnUNet_preprocessed set to: ${nnUNet_preprocessed}"
echo "nnUNet_raw set to: ${nnUNet_raw}"
echo "nnUNet_results set to: ${nnUNet_results}"

# Charger les modules nécessaires (À VÉRIFIER/ADAPTER)
# module purge # Commencer avec un environnement propre
# module load cuda/XXX # Charger le module CUDA si nécessaire pour PyTorch/GPU

# Activer l'environnement virtuel uv (situé dans le dossier nnunet)
VENV_PATH="/home/queberal/Deep_learning_project/nnunet/.venv" # Ajustez si nécessaire
if [ -d "${VENV_PATH}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "Activation de l'environnement uv: ${VENV_PATH}"
    source ${VENV_PATH}/bin/activate
    python --version # Vérifier la version de Python dans l'environnement
    # Assurez-vous que PyTorch avec support GPU est installé et que le trainer custom est accessible
    echo "PyTorch et nnUNet (avec trainer custom) sont supposés être installés dans l'environnement uv."
else
    echo "Erreur: Environnement uv non trouvé à ${VENV_PATH}"
    echo "Assurez-vous d'avoir créé l'environnement avec 'uv venv' dans /home/queberal/Deep_learning_project/nnunet/"
    exit 1
fi

# ----- Vérification PyTorch GPU -----
echo "Vérification de la disponibilité du GPU via PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}'); print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'N/A'}')"
echo "Fin de la vérification PyTorch GPU."
# ----- Fin Vérification -----

# ----- Exécution des étapes nnU-Net (Training) -----

# Définir les variables pour les commandes
PRETRAIN_TASK_ID=2 # NSCLC Dataset ID for pretraining
FINETUNE_TASK_ID=1 # PUMA Dataset ID for finetuning
CONFIG="2d"
FOLD=0
# Trainer class name from your custom implementation
TRAINER_CLASS_NAME="nnUNetTrainerIgnoreIndex"
# Plans identifier used for the pretraining dataset (NSCLC) during preprocessing
PRETRAINING_PLANS_ID="nnUNetPlans_pretrain"
# Plans identifier for the finetuning dataset (PUMA) - usually the default
FINETUNING_PLANS_ID="nnUNetPlans"

# 5. Perform pretraining on NSCLC (Dataset 2) using custom trainer and specific plans
echo "Étape 5: Lancement du pré-entraînement (Task ${PRETRAIN_TASK_ID} - NSCLC) avec ${TRAINER_CLASS_NAME} et plans ${PRETRAINING_PLANS_ID}..."
nnUNetv2_train ${PRETRAIN_TASK_ID} ${CONFIG} ${FOLD} -tr ${TRAINER_CLASS_NAME} -p ${PRETRAINING_PLANS_ID} --npz
if [ $? -ne 0 ]; then
    echo "Erreur lors du pré-entraînement (nnUNetv2_train pour Task ${PRETRAIN_TASK_ID}) avec trainer ${TRAINER_CLASS_NAME} et plans ${PRETRAINING_PLANS_ID}"
    exit 1
fi
echo "Pré-entraînement terminé."

# Construire le chemin vers le meilleur checkpoint du pré-entraînement
# ATTENTION: Vérifiez le nom du dossier 'Dataset002_NSCLC'. Il doit correspondre au dossier créé dans nnUNet_results.
# Le nom est basé sur l'ID et le nom du dataset dans nnUNet_raw.
# Le nom du trainer et des plans doit aussi correspondre exactement.
PRETRAINED_CHECKPOINT_DIR="${nnUNet_results}/Dataset$(printf '%03d' ${PRETRAIN_TASK_ID})_NSCLC/${TRAINER_CLASS_NAME}__${PRETRAINING_PLANS_ID}__${CONFIG}/fold_${FOLD}"
PRETRAINED_WEIGHTS_PATH="${PRETRAINED_CHECKPOINT_DIR}/checkpoint_best.pth"

# Vérifier si le checkpoint existe
if [ ! -f "${PRETRAINED_WEIGHTS_PATH}" ]; then
    echo "Erreur: Checkpoint pré-entraîné non trouvé à ${PRETRAINED_WEIGHTS_PATH}"
    echo "Vérifiez le chemin, notamment le nom du dossier du dataset (Dataset$(printf '%03d' ${PRETRAIN_TASK_ID})_NSCLC), le nom du trainer (${TRAINER_CLASS_NAME}) et l'ID des plans (${PRETRAINING_PLANS_ID})."
    # Lister le contenu pour aider au débogage
    echo "Contenu de ${nnUNet_results}:"
    ls -l ${nnUNet_results}
    # Essayer de lister le contenu du dossier potentiel du dataset pré-entraîné
    echo "Contenu potentiel du dossier du modèle pré-entraîné (Dataset ${PRETRAIN_TASK_ID}):"
    ls -l ${nnUNet_results}/Dataset$(printf '%03d' ${PRETRAIN_TASK_ID})_*
    exit 1
fi
echo "Utilisation du checkpoint pré-entraîné: ${PRETRAINED_WEIGHTS_PATH}"

# 6. Perform finetuning on PUMA (Dataset 1) using standard trainer and default plans, loading pretrained weights
echo "Étape 6: Lancement du finetuning (Task ${FINETUNE_TASK_ID} - PUMA) avec les poids pré-entraînés..."
# Note: We usually don't specify -tr or -p for finetuning unless needed.
# nnU-Net will use the default trainer and plans associated with FINETUNE_TASK_ID (Dataset 1).
nnUNetv2_train ${FINETUNE_TASK_ID} ${CONFIG} ${FOLD} -pretrained_weights ${PRETRAINED_WEIGHTS_PATH} --npz
if [ $? -ne 0 ]; then
    echo "Erreur lors du finetuning (nnUNetv2_train pour Task ${FINETUNE_TASK_ID}) avec poids de ${PRETRAIN_TASK_ID}"
    exit 1
fi
echo "Finetuning terminé."

# nnU-Net conserve par défaut checkpoint_best.pth et checkpoint_final.pth.
# La tâche est considérée comme terminée.

# ----- Fin -----
echo "Entraînement nnU-Net (Pré-entraînement + Finetuning) terminé avec succès."
echo "Le meilleur checkpoint du finetuning devrait se trouver dans le dossier correspondant à Task ${FINETUNE_TASK_ID} dans ${nnUNet_results}"
echo "Date: $(date)"

exit 0
