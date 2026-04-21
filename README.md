# Analyse de Churn Télécom : Déterminisme vs Approches Stochastiques

Ce projet est une application pratique destinée à comparer méthodologiquement la capacité des approches mathématiques déterministes par rapport aux algorithmes d'apprentissage machine stochastiques (arborescents et descente de gradient). Le tout est abordé sous le prisme d'une problématique commerciale concrète : la prédiction du désabonnement des clients (Churn) dans le secteur des télécommunications.

Ce dépôt a été structuré pour répondre à des critères d'évaluation académiques en termes d'analyse exploratoire rigoureuse des données (EDA), d'ingénierie logicielle (pipeline de prétraitement sans fuite de données) et sur le plan de l'interprétabilité éthique algorithmique.

## Structure du Projet

```text
📁 datascience/
│
├── 📄 WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset source issu de Kaggle
├── 📄 X_train.csv, X_test.csv, y_train.csv, y_test.csv # Jeux de données fractionnés après le prétraitement
├── 📄 01_EDA_et_Pretraitement.ipynb        # Phase 1 : Nettoyage, statistiques et visualisations de base
├── 📄 02_Modelisation_Stochastique.ipynb   # Phase 2 : Pipeline d'apprentissage, duel algorithmique et exports
├── 📄 model_deterministe_logreg.pkl        # Modèle entraîné persisté (Régression Logistique)
├── 📄 model_stochastique_random_forest.pkl # Modèle entraîné persisté (Random Forest)
├── 📄 model_stochastique_xgboost.pkl       # Modèle entraîné persisté (XGBoost)
├── 📄 model_stochastique_lightgbm.pkl      # Modèle entraîné persisté (LightGBM)
├── 📄 model_stochastique_sgd.pkl           # Modèle entraîné persisté (SGD Classifier)
├── 📄 app.py                               # Phase 3 : Application Streamlit testant les probabilités en multi-algorithmes
├── 📄 requirements.txt                     # Dépendances nécessaires à l'exécution de l'environnement
│
└── 📁 presentation_assets/                 # Dossier généré automatiquement contenant les assets visuelles des notebooks :
    ├── 1_déséquilibre_classes.png
    ├── 2_correlations_heatmap.png
    ├── 3_pipeline_architecture.txt
    ├── 5_comparaison_roc_auc.png
    └── 6_feature_importance.png
```

## Comment exécuter le projet ?

### 1. Installation de l'environnement
Il est recommandé d'utiliser un environnement virtuel (via `venv` ou `conda`). Une fois votre environnement actif, installez les dépendances requises via la commande :

```bash
pip install -r requirements.txt
```

### 2. Phase Analytique et Modélisation
Afin de générer vos modèles prédictifs et de re-créer les graphiques analytiques, exécutez les carnets de notes Jupyter l'un après l'autre.
Lancez dans votre terminal :

```bash
jupyter notebook
```
- Ouvrez puis exécutez l'entièreté de `01_EDA_et_Pretraitement.ipynb`.
- Ouvrez puis exécutez l'entièreté de `02_Modelisation_Stochastique.ipynb`. (Cette étape génère les `.pkl` requis pour l'application).

### 3. Lancer l'Application de Décision
Une fois vos modèles de Gradient Boosting générés, une interface d'interprétabilité et de classification interactive a été construite sous *Streamlit*. 
Exécutez dans votre terminal :

```bash
streamlit run app.py
```
Un onglet de votre navigateur devrait automatiquement s'ouvrir (habituellement sur `http://localhost:8501`). Saisissez les données d'un profil client pour observer le verdict de résiliation, dicté par une topologie mathématique probabiliste.
