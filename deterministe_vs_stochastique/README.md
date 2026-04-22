# Recherche : Classification Déterministe vs Ensemble Stochastique

## Description
Ce premier axe de notre projet représente le point d'ancrage de notre recherche. Notre objectif est de démontrer de manière très simple pourquoi une équation mathématique stricte (déterministe) échoue souvent à anticiper le comportement humain, qui est par nature complexe et déséquilibré. 

Comme théorisé dès l'introduction de notre rapport et précisé dans le **Chapitre 3.1 (Arbres de décisions stochastiques et Forêts Aléatoires)**, nous avons choisi un terrain d'expérience très concret : la tentative de prédire si un client d'une entreprise Télécom va résilier son abonnement dans les jours à venir (problématique métier du *Churn*).

D'un côté, nous testons un algorithme classique "Déterministe". Il agit comme un juge rigide qui trace une immense ligne droite pour séparer les clients fidèles des clients sur le départ.
De l'autre, nous libérons les approches "Stochastiques" (fondées sur la probabilité). Au lieu d'une ligne stricte, l'ordinateur tire au hasard des centaines de micro-questions pour créer des petits arbres de décisions qui votent tous ensemble. 
Notre logiciel expérimental met au grand jour la victoire écrasante du Hasard : l'intelligence collective des méthodes probabilistes s'adapte aux profils complexes des clients, là où la mathématique classique échoue.

## Structure du Projet
```text
📂 deterministe_vs_stochastique/
 ┣ 📂 data                   # Héberge le dataset brut et les partitions (X_train, y_test...)
 ┣ 📂 models                 # Modèles sérialisés par joblib (.pkl)
 ┣ 📂 presentation_assets    # Matrices de corrélation, Courbes ROC générées
 ┣ 📜 01_EDA_et_Pretraitement.py
 ┣ 📜 02_Modelisation_Stochastique.py
 ┣ 📜 app.py                 # Interface d'exploration de nos résultats
 ┗ 📜 requirements.txt
```

## Guide d'Installation
Assurez-vous de posséder une installation locale de Python 3.9+ ou un environnement virtuel conda.
Ouvrez ce répertoire dans un terminal puis installez les pré-requis de la recherche :
```bash
pip install -r requirements.txt
```

## Guide d'Utilisation
Nos scripts d'analyses (`01_` et `02_`) ont déjà été compilés pour alimenter la base de données de modèles.
Pour interagir avec nos découvertes, démarrez notre interface Streamlit :
```bash
streamlit run app.py
```

### 🧭 Navigation & Expérience sur l'Interface (UI)
Une fois l'application ouverte dans votre navigateur :
1. **Le Sommaire Gauche (Sidebar) :** Utilisez le menu de gauche pour basculer entre l'onglet **"Simulation Interactive"** (pour manipuler concrètement l'IA) et l'onglet **"Analytique & Preuves"** (pour observer les courbes académiques de performance ROC-AUC démontrant nos thèses).
2. **Formulaires de Saisie :** Au sein de la Simulation, vous découvrirez des panneaux de paramètres (Ancienneté, Frais mensuels, Contrats) : ce sont vos variables de contrôle.
3. **Le Comparateur de Probabilités :** Modifiez le profil du client virtuel et validez. Remarquez au centre de l'écran comment la jauge d'intention de résiliation prédite par le modèle "Déterministe Linéaire" échoue très souvent à capter un risque latent qui a pourtant été repéré judicieusement par les modèles "Stochastiques" affichés à côté.
