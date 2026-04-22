# Recherche : Dynamique Macroscopique des Systèmes Multi-Agents (SMA)

## Description
Ce troisième axe de recherche explore un phénomène que l'on appelle "l'émergence". L'objectif est de montrer une chose qui paraît magique en mathématiques : **comment un hasard total à petite échelle finit par créer un ordre prévisible parfait à grande échelle.**

En parfaite adéquation avec le **Chapitre 3.3 (Les Modèles Stochastiques Orientés Agent - SMA)** de notre rapport, nous générons ici un "Micro-Monde" avec de vrais acteurs. Pour cela, nous utilisons l'authentique carte des milliers d'amitiés Facebook de l'Université de Stanford.
L'expérience est très ludique : nous introduisons une "idée" (une rumeur de résiliation d'abonnement ou un virus) dans cette foule d'étudiants connectés. À l'échelle microscopique, c'est le chaos absolu (le stochastique) : personne ne sait ce que va faire un étudiant en particulier. La machine décide de la transmission de l'idée en lançant virtuellement "un dé au hasard" d'une personne à l'autre. 

Pourtant, notre simulation démontre visuellement à l'écran que malgré ce "tirage au sort" chaotique constant de l'ordinateur, la contagion globale de ces jeunes crée invariablement un superbe dessin symétrique (une fameuse "Courbe en Cloche" de propagation pure). C'est la confirmation visuelle ultime que les probabilités ne sont pas un "raté", mais une science mathématique exacte (La Loi des grands nombres).

---

## Méthodologie et Expérimentation (Rapport Technique)

Pour asseoir cette démonstration visuelle d'Émergence Sociale sur des fondations mathématiques incontestables, voici l'anatomie experte du "Moteur Social" développé dans nos laboratoires :

### 1. Acquisition et Topologie Réelle
Pour que la démonstration dépasse le stade du "Jeu", la simulation n'est pas effectuée sur un maillage synthétique ou rectangulaire absolu, mais sur une des bases de données de réseau les plus célèbres au monde :
* **Données Topologiques :** Intégration du projet social du *Stanford Network Analysis Project* (SNAP). 
* **Volume d'Expérience :** Incubation géante de 4039 volontaires (nœuds) et de leurs réelles connexions inter-étudiantes totalisant 88 000 arrêtes sociales asymétriques. Le choix d'un réseau réel hautement irrégulier (avec des super-connectés opposés à des cibles lointaines et isolées) interdit à l'ordinateur de tricher mathématiquement et garantit l'âpreté de nos expériences.

### 2. Algorithmique Stochastique de base (Processus de Markov)
L'intelligence de la simulation ne repose paradoxalement sur aucune équation pré-calculée. Elle se pilote uniquement au travers du croisement de probabilités jetées localement : l'environnement du fameux **Modèle SIR (Sensibles, Infectés, Rétablis/Résiliés)**.
Notre script s'incrémente par "Boucles de Temps" successives (ex. Jour 1, Jour 2...).
À chaque boucle de temps, l'ordinateur visite *chaque* citoyen du graphe, observe ses voisins, et lance en temps réel deux dés conditionnels virtuels (via les fonctions probabilistes de `random`) :
* **Lal variable β (Beta) / Cadrage Stochastique de Contagion :** Si le nœud "A" sain touche un acteur "Infecté/Résiliant", l'ordinateur jette immédiatement ses dés (Ex: Il y a 10% de chances stochastiques d'être effectivement contaminé par ce voisin sur cette boucle).
* **La variable γ (Gamma) / Cadrage Stochastique de Rémission :** Si le nœud est déjà malade, un autre dé tourne en tâche de fond pour savoir si de lui-même ou par l'information sociale, il sera "guéri/vacciné" lors de la journée (La cloche finale).

### 3. Protocole Numérique 
* La dimension d'observation ne s'arrête de figer les probabilités qu'à la limite des 200 "Cycles Historiques". Le système passe par les fonctions de réseau algorithmique `NetworkX` pour gérer l'énorme asymétrie spatiale et le "Time-Saving".
* Il est indispensable, pour démontrer un chaos croissant, que la majorité du graphe débute avec tous ses acteurs en position "X=(Sensible)". Nous introduisons par une injection chirurgicale un unique "Patient Zéro" dans la foule pour lancer toute la dynamique.

### 4. Interprétation Fondamentale (La Preuve Scientifique)
L'interprétation de l'interface graphique en direct prouve empiriquement l'épine dorsale du Mémoire (Le fameux phénomène défendu au **Chapitre 3.3**) :
* **Le désordre total microscopique :** Si l'on posait un capteur d'espionnage sur une seule personne pendant 200 jours, le tirage au sort absolu et multilatéral nous empêcherait de concevoir mathématiquement *qui* elle va contaminer (*Incertitude Radicale* et anti-déterminisme pur sur les profils individuels).
* **La perfection de la matrice macroscopique :** En reculant spatialement de l'individu pour dézoomer brutalement sur le "Total" de la ville ou du réseau, on subit un choc cognitif. L'effroyable collision d'un demi-million de petits calculs probabilistes chaotiques par jour ne donne au final naisance à *aucun* brouillard ! Les courbes observées à notre écran s'élèvent, s'équilibrent et s'opposent avec des rondeurs parfaites. Ce chaos local, non calculé au départ, a accouché sans notre aide d'une macro-symétrie phénoménale de perfection pour la lecture de données en Masse. C'est le triomphe de "L'Emergence de Systèmes Multi-Agents (La Loi des Grands Nombres)".

---

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
