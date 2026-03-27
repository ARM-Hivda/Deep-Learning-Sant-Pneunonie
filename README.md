# 🫁 PneumoScan — Classification de Radiographies Thoraciques par CNN

Pipeline complet de Deep Learning pour la **détection de pneumonie** sur des radiographies thoraciques (Chest X-Ray). Classification binaire `NORMAL / PNEUMONIA` avec interprétabilité Grad-CAM et interface Streamlit.

---

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Dataset](#dataset)
- [Utilisation](#utilisation)
- [Architectures disponibles](#architectures-disponibles)
- [Hyperparamètres](#hyperparamètres)
- [Interface Streamlit](#interface-streamlit)
- [Résultats](#résultats)

---

## 🔬 Aperçu du projet

| Élément | Détail |
|---|---|
| **Tâche** | Classification binaire : NORMAL (0) / PNEUMONIA (1) |
| **Dataset** | Chest X-Ray Images (Kaggle — Paul Mooney) |
| **Architectures** | BaseCNN (from scratch), ResNet-18, DenseNet-121, EfficientNet-B0 |
| **Framework** | PyTorch + TorchVision |
| **Interprétabilité** | Grad-CAM (Selvaraju et al., ICCV 2017) |
| **Interface** | Streamlit (`app.py`) |

---

## 📁 Structure du projet

```
medical-cnn-project/
│
├── app.py                        # Interface Streamlit de démonstration
├── config.yaml                   # Configuration centrale (hyperparamètres)
├── requirements.txt              # Dépendances Python
│
├── src/
│   ├── model.py                  # Architectures CNN (BaseCNN + TransferModel)
│   ├── dataset.py                # Chargement données + transformations
│   ├── train.py                  # Boucle d'entraînement
│   ├── eval.py                   # Évaluation sur le test set
│   ├── gradcam.py                # Visualisation Grad-CAM
│   └── utils.py                  # Fonctions utilitaires
│
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/           # 1 341 images
│       │   └── PNEUMONIA/        # 3 875 images
│       ├── val/
│       │   ├── NORMAL/           # 8 images
│       │   └── PNEUMONIA/        # 8 images
│       └── test/
│           ├── NORMAL/           # 234 images
│           └── PNEUMONIA/        # 390 images
│
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pt         # Meilleur modèle sauvegardé
│   └── figures/
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── error_analysis.png
│
├── reports/                      # Rapports d'évaluation
└── notebooks/
    └── 01_eda.ipynb              # Analyse exploratoire des données
```

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd medical-cnn-project
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Note Windows** : Pour PyTorch avec GPU (CUDA), visiter [pytorch.org](https://pytorch.org/get-started/locally/) pour la commande adaptée à votre version CUDA.

---

## 📦 Dataset

**Source** : [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

```bash
# Via l'API Kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

**Statistiques du dataset :**

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1 341  | 3 875     | 5 216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

> ⚠️ **Déséquilibre de classes** : ratio ≈ 3:1 (PNEUMONIA/NORMAL) géré via `pos_weight=0.33` dans la fonction de perte.

---

## 🚀 Utilisation

### Entraînement

```bash
# BaseCNN (from scratch)
python src/train.py \
    --data_dir data/chest_xray \
    --arch     baseline \
    --epochs   30 \
    --batch    32 \
    --lr       1e-3

# Transfer Learning (ResNet-18)
python src/train.py \
    --data_dir data/chest_xray \
    --arch     resnet18 \
    --epochs   30 \
    --batch    32 \
    --lr       1e-3
```

Ou via `config.yaml` :

```bash
python src/train.py
```

### Évaluation sur le test set

```bash
python src/eval.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --data_dir   data/chest_xray \
    --output     outputs/figures
```

### Visualisation Grad-CAM

```bash
python src/gradcam.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --image      data/chest_xray/test/PNEUMONIA/person1_virus_006.jpeg \
    --output     outputs/figures/gradcam.png
```

### Interface Streamlit

```bash
streamlit run app.py
```

Accéder à : [http://localhost:8501](http://localhost:8501)

---

## 🧠 Architectures disponibles

### BaseCNN (from scratch)

Architecture CNN custom 4 blocs :

```
Input (3×224×224)
  → ConvBlock 1 : Conv(3→32)  + BN + ReLU + MaxPool  → (32×112×112)
  → ConvBlock 2 : Conv(32→64) + BN + ReLU + MaxPool  → (64×56×56)
  → ConvBlock 3 : Conv(64→128)+ BN + ReLU + MaxPool  → (128×28×28)
  → ConvBlock 4 : Conv(128→256)+BN + ReLU + MaxPool  → (256×14×14)
  → Global Average Pooling                            → (256)
  → Dropout(0.5) + FC(256→128) + ReLU
  → Dropout(0.3) + FC(128→1)
  → Sigmoid → P(Pneumonie)
```

### Transfer Learning

| Architecture | Params entraînables (frozen) | Params totaux |
|---|---|---|
| `resnet18` | ~0.5M | 11.2M |
| `densenet121` | ~1M | 8M |
| `efficientnet_b0` | ~1.3M | 5.3M |

Stratégie fine-tuning en 2 phases :
1. **Phase 1** : backbone gelé, entraînement de la tête uniquement
2. **Phase 2** : dégel complet (`model.unfreeze()`), lr réduit

---

## 🔧 Hyperparamètres

Tous les hyperparamètres sont centralisés dans `config.yaml` :

| Paramètre | Valeur | Justification |
|---|---|---|
| `image_size` | `224` | Standard ImageNet / pré-entraînés |
| `epochs` | `30` | Plafond — early stopping arrête avant |
| `batch_size` | `32` | Compromis mémoire GPU / bruit gradient |
| `learning_rate` | `0.001` | Défaut Adam (Kingma & Ba, 2014) |
| `weight_decay` | `0.0001` | Régularisation L2 légère |
| `dropout1` | `0.5` | Fort dropout après GAP (anti-overfitting) |
| `dropout2` | `0.3` | Dropout modéré dans FC |
| `pos_weight` | `0.33` | N_normal/N_pneumo ≈ 1341/3875 |
| `scheduler factor` | `0.5` | Réduction douce du LR (÷2) |
| `scheduler patience` | `3` | Époques avant réduction LR |
| `early_stop patience` | `5` | Scheduler(3) + buffer(2) |
| `gradient clip` | `1.0` | Prévient l'explosion du gradient |

### Fonction de perte

```python
# BCEWithLogitsLoss (numériquement plus stable que Sigmoid + BCE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.33]))
```

### Scheduler

```python
# Réduit lr × 0.5 si val_loss ne s'améliore pas pendant 3 époques
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
```

---

## 🖥️ Interface Streamlit

L'interface `app.py` propose 4 onglets :

| Onglet | Fonctionnalité |
|---|---|
| 🔬 **Analyse d'Image** | Upload, prédiction, barres de probabilité, Grad-CAM |
| 📂 **Analyse par Lot** | Upload multiple, grille de résultats, tableau récapitulatif |
| 🔁 **Pipeline Complet** | Documentation de chaque étape avec justifications |
| ℹ️ **À propos** | Architecture, Grad-CAM, stats dataset |

**Fonctionnalités clés :**
- Chargement du checkpoint via `@st.cache_resource` (une seule fois)
- Seuil de décision ajustable dans la sidebar (0.1 → 0.9)
- Heatmap Grad-CAM interactive (BaseCNN uniquement)
- Design premium dark mode médical

---

## 📊 Résultats

Les courbes d'entraînement et métriques sont sauvegardées dans `outputs/figures/` :

- `training_curves.png` — Loss & Accuracy train/val par époque
- `confusion_matrix.png` — Matrice de confusion sur le test set
- `roc_curve.png` — Courbe ROC avec AUC
- `error_analysis.png` — Visualisation des faux négatifs et faux positifs

**Métriques prioritaires (contexte médical) :**
- **Recall** (sensibilité) : métrique principale — un FN (pneumonie manquée) est critique
- **AUC-ROC** : comparaison inter-architectures indépendante du seuil
- **FNR** (False Negative Rate) : surveillance des cas non détectés

---

## 📚 Références

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016.
- Huang et al., *Densely Connected Convolutional Networks* (DenseNet), CVPR 2017.
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.

---

## ⚠️ Avertissement médical

> Cette application est développée à des fins **éducatives et de recherche uniquement**.
> Les prédictions du modèle ne constituent pas un diagnostic médical et ne remplacent
> en aucun cas l'expertise d'un radiologue qualifié.
> Toute décision médicale doit être prise par un professionnel de santé agréé.
