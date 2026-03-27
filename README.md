# Medical CNN Project — Classification Chest X-Ray (Pneumonie)

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Chest%20X--Ray-green)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 📋 Description

Pipeline complet de **Deep Learning appliqué à l'imagerie médicale**.  
Objectif : classifier des radiographies thoraciques en **NORMAL** ou **PNEUMONIA** à l'aide d'un CNN entraîné from scratch, puis optionnellement par transfert d'apprentissage.

---

## 🗂️ Arborescence

```
medical-cnn-project/
├── config.yaml                  ← Hyperparamètres centralisés
├── requirements.txt             ← Dépendances pip
├── data/
│   └── chest_xray/              ← Dataset Kaggle (à télécharger)
│       ├── train/NORMAL/
│       ├── train/PNEUMONIA/
│       ├── val/NORMAL/
│       ├── val/PNEUMONIA/
│       ├── test/NORMAL/
│       └── test/PNEUMONIA/
├── notebooks/
│   ├── 01_eda.ipynb             ← Exploration des données
│   ├── 02_training_cnn.ipynb   ← Entraînement interactif
│   └── 03_evaluation.ipynb     ← Évaluation et visualisation
├── src/
│   ├── dataset.py               ← Pipeline de données
│   ├── model.py                 ← Architectures CNN
│   ├── train.py                 ← Boucle d'entraînement
│   ├── eval.py                  ← Évaluation finale
│   ├── gradcam.py               ← Interprétabilité Grad-CAM
│   └── utils.py                 ← Utilitaires (seed, métriques…)
├── outputs/
│   ├── checkpoints/             ← best_model.pt
│   └── figures/                 ← Courbes, matrices, heatmaps
└── reports/                     ← Rapport technique final
```

---

## ⚙️ Installation

```bash
# 1. Cloner / copier le projet
cd medical-cnn-project

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 📥 Dataset

Télécharger depuis Kaggle :  
**https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia**

Décompresser dans `data/chest_xray/` en respectant l'arborescence train/val/test.

---

## 🚀 Utilisation rapide

### 1. Entraînement baseline

```bash
python src/train.py \
    --data_dir data/chest_xray \
    --arch     baseline \
    --epochs   30 \
    --batch    32 \
    --lr       1e-3
```

### 2. Évaluation sur le test set

```bash
python src/eval.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --data_dir   data/chest_xray \
    --output     outputs/figures
```

### 3. Grad-CAM (interprétabilité)

```bash
python src/gradcam.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --image      data/chest_xray/test/PNEUMONIA/person1_virus_006.jpeg \
    --output     outputs/figures/gradcam.png
```

### 4. Transfer Learning (extension)

```bash
python src/train.py --arch resnet18 --lr 1e-4 --epochs 20
```

---

## 📊 Métriques suivies

| Métrique      | Description                                          |
|---------------|------------------------------------------------------|
| Accuracy      | % de prédictions correctes                           |
| Recall        | Taux de détection des pneumonies (**prioritaire**)   |
| Precision     | Fiabilité des alertes positives                     |
| F1-score      | Équilibre Precision/Recall                           |
| AUC-ROC       | Performance indépendante du seuil                   |

---

## 📅 Plan de 10 jours

| Jour | Activités |
|------|-----------|
| J1   | Cadrage, installation, téléchargement dataset |
| J2   | EDA, comptage classes, visualisation, détection corrompues |
| J3   | Pipeline chargement, resize, normalisation, premiers batchs |
| J4   | Implémentation CNN baseline, boucle d'entraînement |
| J5   | Premier entraînement complet, courbes train/val |
| J6   | Ajustements : batch size, LR, dropout, augmentation |
| J7   | Second entraînement, comparaison baseline |
| J8   | Évaluation finale : confusion, accuracy, recall, F1, AUC |
| J9   | Analyse qualitative des erreurs, Grad-CAM (si temps) |
| J10  | Présentation finale / rapport technique |

---

## 📦 Livrables

1. ✅ Code source fonctionnel (`src/`, `notebooks/`)
2. ✅ Modèle entraîné (`outputs/checkpoints/best_model.pt`)
3. ✅ Résultats : courbes, matrice de confusion, métriques
4. ✅ Rapport technique (`reports/`)

---

## 🔗 Références

- Kaggle — [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- LeCun, Bengio, Hinton — *Deep Learning*, Nature, 2015
- Selvaraju et al. — *Grad-CAM*, ICCV 2017
