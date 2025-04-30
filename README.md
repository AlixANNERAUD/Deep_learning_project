<h1 align="center">Deep Learning Project pour la Segmentation et Classification d'Images Médicales</h1>

<p align="center">
Un projet d'application de techniques de deep learning pour la segmentation et la classification de cellules et tissus dans des images médicales.
</p>

## 📋 Vue d'ensemble du projet

Ce projet implémente plusieurs modèles de deep learning pour la segmentation et la classification d'images médicales, spécifiquement focalisés sur les données histopathologiques. Il utilise des frameworks de pointe comme nnUNet et CellViT pour traiter efficacement ces tâches.

Le projet est structuré autour de plusieurs modules qui fonctionnent ensemble pour traiter les données brutes, entraîner les modèles et analyser les résultats.

## 🔧 Installation

Le projet utilise [uv](https://github.com/astral-sh/uv), un installateur et gestionnaire d'environnements Python rapide, pour gérer les dépendances.

```bash
# Installation générale
cd backend
uv lock
uv sync --extra cpu | apple | cu124
```

Chaque sous-module dispose également de son propre environnement qui peut être configuré grâce à uv.

## 📂 Structure du projet

Le projet est organisé en plusieurs modules principaux:

### 1. nnUNet

[nnUNet](https://github.com/MIC-DKFZ/nnUNet) est un framework de segmentation d'images médicales auto-configurant. Dans notre projet, il est utilisé pour la segmentation automatique des tissus et des noyaux cellulaires.

#### Scripts principaux:

-   `prepare_dataset_puma_nnunet.py` : Prépare les données PUMA pour l'entraînement avec nnUNet
-   `prepare_dataset_nsclc_nnunet.py` : Prépare les données NSCLC pour l'entraînement
-   `preprocess_nnunet.slurm` : Script SLURM pour prétraiter les données
-   `train_nnunet.slurm` : Script SLURM pour l'entraînement du modèle nnUNet

### 2. CellViT++

[CellViT++](https://github.com/TIO-IKIM/CellViT) est un framework de vision par transformateur pour la segmentation et la classification des cellules dans des images histologiques.

#### Scripts principaux:

-   `prepare_puma_for_cellvit.py` : Prépare les données PUMA pour l'entraînement avec CellViT
-   `prepare_puma_for_cellvit_segmentation.py` : Prépare les données pour la tâche de segmentation
-   `train_cellvit_classifier.slurm` : Script SLURM pour l'entraînement du classificateur CellViT
-   `training-configuration.yaml` : Configuration pour l'entraînement du modèle

### 3. HoVer-Net

[HoVer-Net](https://github.com/vqdang/hover_net) est une architecture pour la segmentation des noyaux cellulaires et la classification des types de cellules.

#### Scripts principaux:

-   `code/preprocess_data.py` : Prétraitement des données pour HoVer-Net
-   `code/train_hovernet.slurm` : Script SLURM pour l'entraînement du modèle
-   `run_infer.py` : Script pour l'inférence avec le modèle entraîné
-   `run_train.py` : Script pour lancer l'entraînement

### 4. Diaporama et Visualisation

Le dossier `diaporama/` contient des notebooks et documents pour présenter et visualiser les résultats du projet:

-   `diaporama.ipynb` / `diaporama.md` / `diaporama.pdf` : Présentations du projet sous différents formats
-   `wandbinfo.ipynb` : Notebook pour visualiser les logs d'expériences avec Weights & Biases

## 🗃️ Données

Le projet utilise principalement des datasets histopathologiques:

-   Images de ROI (Regions of Interest) au format TIF
-   Annotations au format GeoJSON pour les noyaux et les tissus
-   Données contextuelles pour une analyse plus approfondie

Les données se trouvent dans le dossier `dataset/` et sont organisées par type d'annotation.

## 🚀 Utilisation des modèles

### nnUNet

```bash
# Préparation des données PUMA pour nnUNet
cd nnunet
python prepare_dataset_puma_nnunet.py

# Lancer l'entraînement avec SLURM
sbatch train_nnunet.slurm
```

### CellViT++

```bash
# Préparation des données pour CellViT
cd cellvit_plus_plus
python prepare_puma_for_cellvit.py

# Lancer l'entraînement du classificateur
sbatch train_cellvit_classifier.slurm
```

### HoVer-Net

```bash
# Prétraitement des données
cd code
python preprocess_data.py

# Lancer l'entraînement avec SLURM
sbatch train_hovernet.slurm
```

## 🔄 Gestion des environnements avec uv

Chaque sous-module du projet a son propre environnement virtuel géré par uv:

```bash
# Installer l'environnement d'un sous-module spécifique
cd [nom_du_module]
uv lock
uv sync
```

Les fichiers `uv.lock` dans chaque sous-répertoire spécifient les versions exactes des dépendances pour garantir la reproductibilité.

## 📝 Note sur les submodules

Le projet utilise plusieurs submodules Git:

1. **CellViT-plus-plus**: Implémentation complète de CellViT avec des outils d'annotation et de visualisation
2. **hover_net**: Implémentation de HoVer-Net pour la segmentation des noyaux cellulaires
3. **nnUNet**: Framework de segmentation d'images médicales

Ces submodules sont intégrés au projet principal et peuvent être mis à jour avec:

```bash
git submodule update --init --recursive
```

## 🛠 Customisations

Des classes et fonctions personnalisées ont été développées pour adapter les frameworks existants à nos besoins spécifiques:

-   `nnunet_custom_classes/`: Extensions personnalisées pour nnUNet
-   Configurations spécifiques pour chaque framework dans leurs dossiers respectifs

## 🤝 Contributeurs

Ce projet est développé dans le cadre d'un cours à l'INSA Rouen.

## 📄 Licence

Ce projet est distribué sous licence spécifiée dans le fichier LICENSE à la racine du projet.
