# Recherche : Stratégies d'Optimisation des Espaces de Modèles

## Description
Ce deuxième axe de recherche montre comment on "règle" une Intelligence Artificielle. Un peu comme on accorde les cordes d'une guitare avant un concert, une machine a besoin de trouver les réglages parfaits (appelés hyperparamètres) pour être performante.

Comme expliqué dans le **Chapitre 2 (Optimisation)** de notre rapport de thèse, chercher ces réglages parfaits de manière rigide ("Déterministe") est en réalité un cauchemar informatique. L'approche classique (le fameux *Grid Search*) consiste bêtement à essayer toutes les combinaisons possibles sur Terre, une par une de manière stricte. Cela prend un temps énorme, rendant la machine presque incapable de finir son calcul en direct.

Pour contourner ce mur d'heures de calculs, ce module démontre en direct la supériorité de l'approche "Stochastique Bayésienne" (le hasard intelligent). L'idée est astucieuse : l'ordinateur *teste au hasard* quelques réglages, se rend compte très vite des zones qui marchent mal, et "oriente" ses prochains tirs au hasard uniquement vers ce qui a l'air de fonctionner (via la technologie *Optuna*). 
L'expérience menée ici chronomètre cette course devant le jury : elle prouve de manière flagrante que la méthode probabiliste écrase l'approche classique en trouvant le réglage parfait en un temps record.

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
