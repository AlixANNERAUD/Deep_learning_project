#!/bin/bash

#SBATCH --job-name=nnunet_preprocess # Nom du job
#SBATCH --output=nnunet_preprocess_%j.out # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=nnunet_preprocess_%j.err  # Fichier de sortie d'erreur (%j = ID du job)
#SBATCH -p mesonet                  # Nom de la partition (obligatoire)
#SBATCH --account=m25031            # !! REMPLACEZ m2xxxx par votre ID de projet !! (obligatoire)
#SBATCH --nodes=1                   # Nombre de nœuds requis
#SBATCH --ntasks-per-node=1         # Nombre de tâches (processus) par nœud
#SBATCH --cpus-per-task=16          # Nombre de cœurs CPU par tâche (ajustable)
#SBATCH --mem=64G                   # Mémoire RAM requise par nœud (ajustable)
#SBATCH --time=02:00:00             # Temps maximum d'exécution (HH:MM:SS, ajustable)
#SBATCH --gres=gpu:0                # Pas de GPU demandé pour le pré-traitement

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
# module load jq # Décommentez si jq n'est pas disponible par défaut

# Activer l'environnement virtuel uv (situé dans le dossier nnunet)
VENV_PATH="/home/queberal/Deep_learning_project/nnunet/.venv" # Ajustez si nécessaire
if [ -d "${VENV_PATH}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "Activation de l'environnement uv: ${VENV_PATH}"
    source ${VENV_PATH}/bin/activate
    python --version # Vérifier la version de Python dans l'environnement
    echo "PyTorch est supposé être déjà installé dans l'environnement uv."
else
    echo "Erreur: Environnement uv non trouvé à ${VENV_PATH}"
    echo "Assurez-vous d'avoir créé l'environnement avec 'uv venv' dans /home/queberal/Deep_learning_project/nnunet/"
    exit 1
fi

# ----- Exécution des étapes nnU-Net (Logique Pré-entraînement) -----
echo "Lancement des étapes de pré-traitement nnU-Net (Logique Pré-entraînement)..."

FINETUNING_DATASET_ID=1 # PUMA
PRETRAINING_DATASET_ID=2 # NSCLC
FINETUNING_PLANS_ID="nnUNetPlans" # Plans par défaut pour PUMA
PRETRAINING_PLANS_ID="nnUNetPlans_pretrain" # Identifiant personnalisé pour les plans transférés sur NSCLC
CONFIG_KEY="2d" # Configuration à utiliser/modifier

# 1. Planification et pré-traitement pour Dataset de Finetuning (PUMA, ID 1)
echo "Étape 1: Planification et pré-traitement pour Dataset ${FINETUNING_DATASET_ID} (Finetuning)..."
nnUNetv2_plan_and_preprocess -d ${FINETUNING_DATASET_ID} --verify_dataset_integrity
if [ $? -ne 0 ]; then echo "Erreur lors de nnUNetv2_plan_and_preprocess pour Dataset ${FINETUNING_DATASET_ID}"; exit 1; fi
# S'assurer que le dossier existe (peut avoir été renommé manuellement)
FINETUNING_PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset$(printf '%03d' ${FINETUNING_DATASET_ID})_PUMA" # Utilise le nom PUMA comme convenu
if [ ! -d "${FINETUNING_PREPROCESSED_DIR}" ]; then
    echo "Erreur: Dossier pré-traité ${FINETUNING_PREPROCESSED_DIR} non trouvé après l'étape 1."
    exit 1
fi
echo "Étape 1 terminée."

# 2. Extraction du fingerprint pour Dataset de Pré-entraînement (NSCLC, ID 2)
echo "Étape 2: Extraction du fingerprint pour Dataset ${PRETRAINING_DATASET_ID} (Pré-entraînement)..."
nnUNetv2_extract_fingerprint -d ${PRETRAINING_DATASET_ID}
if [ $? -ne 0 ]; then echo "Erreur lors de nnUNetv2_extract_fingerprint pour Dataset ${PRETRAINING_DATASET_ID}"; exit 1; fi
echo "Étape 2 terminée."

# 3. Modification des plans pour Dataset de Finetuning (PUMA, ID 1) AVANT transfert
echo "Étape 3: Modification du fichier ${FINETUNING_PLANS_ID}.json pour Dataset ${FINETUNING_DATASET_ID}..."
PLANS_FILE="${FINETUNING_PREPROCESSED_DIR}/${FINETUNING_PLANS_ID}.json"

if [ -f "${PLANS_FILE}" ]; then
    # Vérifier si jq est disponible
    if command -v jq &> /dev/null; then
        echo "Modification de ${PLANS_FILE} avec jq..."
        # Utilisation de --arg pour passer la clé de configuration de manière plus sûre
        jq --arg key "${CONFIG_KEY}" \
           '.configurations[$key].batch_size = 12 | .configurations[$key].patch_size = [512, 512]' \
           "${PLANS_FILE}" > "${PLANS_FILE}.tmp" && mv "${PLANS_FILE}.tmp" "${PLANS_FILE}"
        if [ $? -ne 0 ]; then
            echo "Erreur lors de la modification de ${PLANS_FILE} avec jq."
            rm -f "${PLANS_FILE}.tmp" # Nettoyer le fichier temporaire en cas d'erreur
            exit 1
        fi
        echo "Fichier ${PLANS_FILE} modifié: batch_size=12, patch_size=[512, 512] pour la configuration '${CONFIG_KEY}'."
    else
        echo "Erreur: La commande 'jq' n'a pas été trouvée. Impossible de modifier ${PLANS_FILE} automatiquement."
        echo "Veuillez modifier manuellement le fichier ${PLANS_FILE} avant de continuer ou installer/charger jq."
        exit 1
    fi
else
    echo "Erreur: Le fichier de plans ${PLANS_FILE} n'a pas été trouvé."
    exit 1
fi
echo "Étape 3 terminée."

# 4. Transfert des plans MODIFIÉS du Dataset Finetuning (1) vers Pré-entraînement (2) avec nouvel ID
echo "Étape 4: Transfert des plans ${FINETUNING_PLANS_ID} (modifiés) de Dataset ${FINETUNING_DATASET_ID} vers Dataset ${PRETRAINING_DATASET_ID} avec l'ID ${PRETRAINING_PLANS_ID}..."
nnUNetv2_move_plans_between_datasets -s ${FINETUNING_DATASET_ID} -t ${PRETRAINING_DATASET_ID} -sp ${FINETUNING_PLANS_ID} -tp ${PRETRAINING_PLANS_ID}
if [ $? -ne 0 ]; then echo "Erreur lors de nnUNetv2_move_plans_between_datasets"; exit 1; fi
echo "Étape 4 terminée."

# 5. Préparer le dossier pré-traité pour Dataset de Pré-entraînement (NSCLC, ID 2)
echo "Étape 5: Création du dossier pré-traité et copie de dataset.json pour Dataset ${PRETRAINING_DATASET_ID}..."
# Déterminer le nom du dossier raw (peut varier)
DATASET2_RAW_NAME=$(basename $(find ${nnUNet_raw} -maxdepth 1 -type d -name "Dataset$(printf '%03d' ${PRETRAINING_DATASET_ID})_*"))
if [ -z "${DATASET2_RAW_NAME}" ]; then
    echo "Erreur: Impossible de trouver le dossier raw pour Dataset ID ${PRETRAINING_DATASET_ID} dans ${nnUNet_raw}"
    exit 1
fi
DATASET2_RAW_DIR="${nnUNet_raw}/${DATASET2_RAW_NAME}"
# Utiliser le même nom pour le dossier pré-traité
DATASET2_PREPROCESSED_DIR="${nnUNet_preprocessed}/${DATASET2_RAW_NAME}"

mkdir -p "${DATASET2_PREPROCESSED_DIR}"
if [ -f "${DATASET2_RAW_DIR}/dataset.json" ]; then
    cp "${DATASET2_RAW_DIR}/dataset.json" "${DATASET2_PREPROCESSED_DIR}/"
    echo "dataset.json copié vers ${DATASET2_PREPROCESSED_DIR}."
else
    echo "Erreur: ${DATASET2_RAW_DIR}/dataset.json non trouvé."
    exit 1
fi
echo "Étape 5 terminée."

# 6. Pré-traitement pour Dataset de Pré-entraînement (NSCLC, ID 2) en utilisant les plans transférés
echo "Étape 6: Pré-traitement pour Dataset ${PRETRAINING_DATASET_ID} avec les plans ${PRETRAINING_PLANS_ID}..."
nnUNetv2_preprocess -d ${PRETRAINING_DATASET_ID} -plans_name ${PRETRAINING_PLANS_ID} -c ${CONFIG_KEY}
if [ $? -ne 0 ]; then echo "Erreur lors de nnUNetv2_preprocess pour Dataset ${PRETRAINING_DATASET_ID}"; exit 1; fi
echo "Étape 6 terminée."


# ----- Fin -----
echo "Pré-traitement nnU-Net (Logique Pré-entraînement) terminé avec succès."
echo "Date: $(date)"

exit 0