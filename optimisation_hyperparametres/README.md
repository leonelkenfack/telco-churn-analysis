# Recherche : Stratégies d'Optimisation des Espaces de Modèles

## Description
Ce deuxième axe de recherche montre comment on "règle" une Intelligence Artificielle. Un peu comme on accorde les cordes d'une guitare avant un concert, une machine a besoin de trouver les réglages parfaits (appelés hyperparamètres) pour être performante.

Comme expliqué dans le **Chapitre 2 (Optimisation)** de notre rapport de thèse, chercher ces réglages parfaits de manière rigide ("Déterministe") est en réalité un cauchemar informatique. L'approche classique (le fameux *Grid Search*) consiste bêtement à essayer toutes les combinaisons possibles sur Terre, une par une de manière stricte. Cela prend un temps énorme, rendant la machine presque incapable de finir son calcul en direct.

Pour contourner ce mur d'heures de calculs, ce module démontre en direct la supériorité de l'approche "Stochastique Bayésienne" (le hasard intelligent). L'idée est astucieuse : l'ordinateur *teste au hasard* quelques réglages, se rend compte très vite des zones qui marchent mal, et "oriente" ses prochains tirs au hasard uniquement vers ce qui a l'air de fonctionner (via la technologie *Optuna*). 
L'expérience menée ici chronomètre cette course devant le jury : elle prouve de manière flagrante que la méthode probabiliste écrase l'approche classique en trouvant le réglage parfait en un temps record.

---

## Méthodologie et Expérimentation (Rapport Technique)

Afin d'étayer cette vaste démonstration chronométrée, voici la structure formelle de l'environnement d'optimisation qui tourne en arrière-plan de ce pôle de recherche :

### 1. Prétraitement et Cadrage de l'Espace Matériel
* **Assise des données :** Le module recharge d'exploitation les données du système Telco (fort déséquilibre des classes cibles), traitées via notre matrice de conversion (StandardScaler et Encodages catégoriels).
* **Bridage temporel (Le Mur de la Complexité) :** Le principal problème des algorithmes déterministes d'exploration complète est l'explosion factorielle (Leurs jours complets de calcul CPU !). Nous réduisons donc volontairement les profondeurs de l'algorithme "Grid Search" à une "grille" modeste mais tout de même handicapante, afin de pouvoir générer les résultats en un temps compatible avec une présentation de laboratoire.

### 2. Espace de Recherche Vectoriel
L'expérimentation porte sur l'optimisation des hyperparamètres des algorithmes d'ensemble stochastiques ultra-tous-terrains étudiés précédemment (LightGBM, Random Forest). 
L'arène de recherche met en jeu trois leviers vitaux pour la machine :
* Le `learning_rate` : Ou "pas d'apprentissage" (Saut mathématique pour la descente de Gradient).
* La `max_depth` (et min_samples_leaf) : La contrainte de simplification de l'algorithme, rempart absolu contre le Surapprentissage.
* Le `n_estimators` : La masse du système évaluatif.

### 3. Protocole Comparatif (Le Banc d'Essai)
L'architecture Python oppose une boucle sur trois philosophies totalement antagonistes :
1. **`GridSearchCV` (Le Stricte/Déterministe) :** Validation mécanique brute. L'IA va créer autant de modèles que de déclinaisons possibles mathématiquement au sein de nos listes pré-définies (Quadrillage temporel désastreux).
2. **`RandomizedSearchCV` (Le Hasard Chaotique) :** Validation Monte-Carlo. La machine procède par essais-erreurs pur et borné, tirant au sort des lots de variables, en espérant "toucher le gros lot" statistique plus facilement que l'approche rigide.
3. **`Optuna - TPE` (Le Hasard Stochastique Informé) :** Processus Bayésien par excellence (Tree-structured Parzen Estimator). La machine tire un premier coup au hasard, étudie son "Loss", et *redimensionne logiquement son futur pôle de hasard* lors du tirage suivant. Elle traque frénétiquement la courbe ascendante.

### 4. Interprétation Fondamentale (La Preuve Scientifique)
La conclusion chiffrée, mise en forme par notre bulle visuelle interactive, entérine officiellement l'hypothèse de recherche :
* **Un désastre Déterministe :** Coincée sur l'extrême droite du graphique, l'approche *Grid Search* affiche un temps de traitement monumental pour un score parfois en deçà. 
* **L'aboutissement de la recherche Stochastique :** Sur l'extrémité gauche se regroupe la grappe de pointe (Optuna). Grâce au concept bayésien guidé, le ratio d'effort s'effondre. Moins de tentatives processeurs pour un F1-Score et ROC-AUC quasi parfait : démontrer que l'incertitude bayésienne n'est pas "floue", mais est le pinacle actuel de l'Ingénierie IA asymétrique moderne.

---

## Structure du Projet
```text
📂 optimisation_hyperparametres/
 ┣ 📂 data                   # Fichier CSV contenant les résultats de nos compilations chronométrées
 ┣ 📂 models                 # (Réservé au transfert expérimental)
 ┣ 📂 presentation_assets    # Visualisation du Duel "Temps passé vs Résultat ROC-AUC"
 ┣ 📜 01_Comparaison_Optimiseurs.py
 ┣ 📜 app.py                 # Tableau de bord analytique de l'optimisation
 ┗ 📜 requirements.txt
```

## Guide d'Installation
Un environnement Python exécutant les packages avancés d'optimisation (Optuna) est requis :
```bash
pip install -r requirements.txt
```

## Guide d'Utilisation
La collecte de nos benchmarks massifs gérés par `01_Comparaison_Optimiseurs.py` a déjà été effectuée en amont. 
Visualisez ces rapports de recherche en initiant le composant frontal :
```bash
streamlit run app.py
```

### 🧭 Navigation & Expérience sur l'Interface (UI)
Une fois l'application chargée, l'interface Web affichera d'emblée la restitution visuelle de la compétition de nos algorithmes.
1. **Zone Chronométrique (Métriques) :** En haut de page, nous avons disposé fièrement des compteurs d'efficience. Vous ferez l'implacable constat de l'écrasement des durées de recherche par notre processus bayésien (optuna) par rapport aux heures incalculables exigées de la boucle Grid Search.
2. **Le Graphe Multidimensionnel :** Observez le grand graphique central croisant le Temps (Axe X) et la Performance AUC (Axe Y). La navigation permet au jury de placer la souris sur les "points du graphique" (points de convergence) pour lire les métriques techniques exactes validant que le point optimal (en haut à gauche) est bel et bien le fruit des méthodes stochastiques sur l'un de nos algorithmes dominants (comme LightGBM).
