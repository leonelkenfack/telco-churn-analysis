# Recherche : Dynamique Macroscopique des Systèmes Multi-Agents (SMA)

## Description
Ce troisième axe de recherche explore un phénomène que l'on appelle "l'émergence". L'objectif est de montrer une chose qui paraît magique en mathématiques : **comment un hasard total à petite échelle finit par créer un ordre prévisible parfait à grande échelle.**

En parfaite adéquation avec le **Chapitre 3.3 (Les Modèles Stochastiques Orientés Agent - SMA)** de notre rapport, nous générons ici un "Micro-Monde" avec de vrais acteurs. Pour cela, nous utilisons l'authentique carte des milliers d'amitiés Facebook de l'Université de Stanford.
L'expérience est très ludique : nous introduisons une "idée" (une rumeur de résiliation d'abonnement ou un virus) dans cette foule d'étudiants connectés. À l'échelle microscopique, c'est le chaos absolu (le stochastique) : personne ne sait ce que va faire un étudiant en particulier. La machine décide de la transmission de l'idée en lançant virtuellement "un dé au hasard" d'une personne à l'autre. 

Pourtant, notre simulation démontre visuellement à l'écran que malgré ce "tirage au sort" chaotique constant de l'ordinateur, la contagion globale de ces jeunes crée invariablement un superbe dessin symétrique (une fameuse "Courbe en Cloche" de propagation pure). C'est la confirmation visuelle ultime que les probabilités ne sont pas un "raté", mais une science mathématique exacte (La Loi des grands nombres).

## Structure du Projet
```text
📂 simulation_systeme_multi_agents/
 ┣ 📂 data                   # Dataset de Stanford (facebook_combined.txt.gz) et tableurs historiques
 ┣ 📂 models                 # (Inusité en simulation virale)
 ┣ 📂 presentation_assets    # Preuves en courbes png persistées de l'émergence
 ┣ 📜 01_Contagion_SMA.py
 ┣ 📜 app.py                 # Interface Streamlit d'expérimentation en simulation temps-réel
 ┗ 📜 requirements.txt
```

## Guide d'Installation
L'exécution de la physique des réseaux complexes de Stanford exige les modules suivants :
```bash
pip install -r requirements.txt
```

## Guide d'Utilisation
La puissance de ce scénario du projet s'observe de manière vivante ! Connectez-vous à la modélisation à l'aide de :
```bash
streamlit run app.py
```

### 🧭 Navigation & Expérience sur l'Interface (UI)
Ce pôle est extrêmement ludique et interactif. Il requiert un temps de chargement initial minimal de 2 à 4 secondes, le temps pour notre serveur d'absorber en cache la base de Stanford (88,000 relations sociales !).
1. **Le Laboratoire Latéral (Menu de gauche) :** Il met à votre disposition les seules variables de création du hasard que le système nécessite pour fonctionner. Modifiez le curseur d'infection stochastique (la chance de contamination β) ou la résilience systémique (γ) d'une population.
2. **Le Déclencheur Mondial :** Validez vos "jetés de dés" initiaux en cliquant sur `Simuler la dynamique Sociale`.
3. **Le Phénomène d'Émergence :** L'écran central générera alors son graphique "en temps réel", itération par itération. Un chaos informe de lignes naîtra dans les premières fractions de secondes, avant que le lecteur ne perçoive sans ambiguïté la création magistrale, fluide et inarrêtable d'une "courbe en cloche" mathématiquement parfaite de bout en bout (La fameuse courbe SIR épidémiologique d'émergence) venant cautionner la pertinence de la simulation multi-agents !
