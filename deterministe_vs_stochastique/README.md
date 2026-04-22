# Recherche : Classification Déterministe vs Ensemble Stochastique

## Description
Ce premier axe de notre projet représente le point d'ancrage de notre recherche. Notre objectif est de démontrer de manière très simple pourquoi une équation mathématique stricte (déterministe) échoue souvent à anticiper le comportement humain, qui est par nature complexe et déséquilibré. 

Comme théorisé dès l'introduction de notre rapport et précisé dans le **Chapitre 3.1 (Arbres de décisions stochastiques et Forêts Aléatoires)**, nous avons choisi un terrain d'expérience très concret : la tentative de prédire si un client d'une entreprise Télécom va résilier son abonnement dans les jours à venir (problématique métier du *Churn*).

D'un côté, nous testons un algorithme classique "Déterministe". Il agit comme un juge rigide qui trace une immense ligne droite pour séparer les clients fidèles des clients sur le départ.
De l'autre, nous libérons les approches "Stochastiques" (fondées sur la probabilité). Au lieu d'une ligne stricte, l'ordinateur tire au hasard des centaines de micro-questions pour créer des petits arbres de décisions qui votent tous ensemble. 
Notre logiciel expérimental met au grand jour la victoire écrasante du Hasard : l'intelligence collective des méthodes probabilistes s'adapte aux profils complexes des clients, là où la mathématique classique échoue.

---

## Méthodologie et Expérimentation (Rapport Technique)

Afin d'offrir une complète transparence sur les mécanismes de cette expérience fondatrice comparant les approches déterministes et probabilistes, voici l'anatomie exacte du flux de recherche implémenté dans nos scripts :

### 1. Prétraitement des Données (Data Processing)
* **Dataset d'origine :** Utilisation de la base transactionnelle classique "Telco Customer Churn" totalisant 7043 profils clients annotés.
* **Nettoyage et Transformation :** Rectification des variables corrompues (remplacement des espaces vides par des 0 numériques absolus pour gérer les charges totales réelles).
* **Encodage & Normalisation (Pipeline analytique) :** Pour assurer le fonctionnement mathématique de nos modèles déterministes de contrôle, nous implémentons un gestionnaire `ColumnTransformer`. Les variables continues (Frais) sont normalisées mathématiquement via un `StandardScaler` (distribution centrée réduite). Par précaution, les variables qualifiantes (Types de contrats, Sexe) sont converties numériquement de manière binaire (via `OneHotEncoder`).
* **Séparation Stochastique Stratifiée :** La variable cible (Résiliation visée ou non) étant structurellement déséquilibrée (~73% de clients stables contre ~27% de démissionnaires), la coupure des données d'apprentissage (Train/Test) applique un *coefficient de stratification* mathématique. Cela garantit de n'introduire aucun biais conjoncturel dans la distribution de validation.

### 2. Architecture et Modélisations Mathématiques
L'expérience a nécessité le chargement de cinq approches analytiques concurrentes.
* **Le Référentiel (Déterministe Linéaire) :** Il s'agit d'une **Régression Logistique** classique. Ce bloc de base résout froidement une seule et grande équation à variables multiples bornée par fonction Sigmoïde de façon continue.
* **L'Arsenal Stochastique (Apprentissage d'Ensemble) :** Conçu pour opposer collectivement la brutalité de la théorie des probabilités à l'équation précédente.
   * **Random Forest (Bagging probabiliste) :** Génération de centaines d'arbres de choix en échantillonnant aléatoirement certaines lignes client ET certaines colonnes descriptives. Cela simule une multitude de points de vue humains très diversifiés ("Intelligence via le Hasard").
   * **XGBoost & LightGBM (Gradient Boosting) :** Algorithmes itératifs ultra-compétitifs dont la logique pousse à corriger les résidus d'erreurs stochastiques successives. Le modèle suivant grandit sur les erreurs tirées du modèle précédent.
   * **SGD Classifier :** Exploration du plan d'optimisation via la force de la Descente de Gradient Stochastique.

### 3. Protocole d'Entraînement Algorithmique
La globalité de ces engrenages est assujettie à une constance universelle de "Graines d'initialisations" stochastiques (`random_state=42`) pour certifier la stricte reproductibilité académique des opérations devant le jury. 
Face au déséquilibre profond de nos classes (le cas très concret du comportement naturel humain face au désabonnement), l'évaluation finale du laboratoire occulte catégoriquement la mesure simpliste du score d'Exactitude ("Accuracy"), très défaillante dans ce cas de figure, au seul et unique bénéfice de la métrique ROC-AUC, la force scientifique par excellence.

### 4. Interprétation Fondamentale (La Preuve Scientifique)
En couplant vos résultats aux analytiques affichées sur l'interface numérique :
* **Dérive Dimensionnelle des Courbes ROC-AUC :** Le modèle logistique trace une limite décisionnelle souvent imparfaite et aveugle lorsque les motifs comportementaux dérivent de l'équation absolue. La victoire des modèles stochastiques (LightGBM et le Random Forest en tête de cordée) capte cette complexité mathématique des profils à très forte dimension d'une façon incontestable, en générant des aires sous courbes nettement plus vastes.
* **Le Graphe Explicatif (L'Extraction d'Information) :** C'est le coup de grâce à l'aspect Déterministe. Les représentations extraites démontrent clairement le fonctionnement de la logique "Ensembliste". L'ordinateur déterministe attribue basiquement et bêtement un poids coefficientiel statique (Exemple : Si vous avez un incident internet, c'est X d'intention de résiliation, peu importe qui vous êtes). Le modèle Stochastique fait tout autre chose, il isole des **Gains d'Informations croisés et dynamiques**. La probabilité modifie elle-même son raisonnement et priorise les frais financiers pour un segment d'usagers, tandis que sur la même table, elle interroge le type de contrat pour un tout autre abonné ! C'est ce concept de non-linéarité comportementale qui forge la puissance de l'Intelligence Collective Probabiliste mis au jour.

---

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
