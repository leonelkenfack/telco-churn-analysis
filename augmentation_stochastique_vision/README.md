# Recherche : Régularisation d'Images par le Hasard Temporel

## Description
Cette expérimentation finale porte sur l'Intelligence Artificielle de reconnaissance d'images. Notre but est de rendre un concept informatique complexe totalement accessible : **comprendre pourquoi une Intelligence Artificielle "apprend mal" si on ne l'entraîne pas avec un peu de hasard.**

Comme abordé dans notre rapport de recherche au niveau du **Chapitre 3.2 (Stratégies d'Augmentation Stochastique de Données)**, une machine trop rigide (déterministe) a un grand défaut : elle fait du "par cœur" (phénomène appelé surapprentissage). Si on lui donne à apprendre 1000 photos parfaites de vêtements, elle va simplement les mémoriser comme un robot, mais paniquera totalement si on lui donne demain une vraie photo d'un client légèrement de travers ou floue. 

Pour protéger l'Intelligence Artificielle de cette mémorisation "bête", ce scénario démontre sur nos écrans l'approche Stochastique. L'idée est simple : on va volontairement demander au hasard de "brouiller" les images. À chaque fois qu'on montre un vêtement à l'Intelligence Artificielle lors de sa formation, une probabilité va incliner la photo, zoomer ou la retourner l'image (effet miroir). 
L'ordinateur ne voyant techniquement plus la même image pure : cela l'empêche de tricher et de mémoriser. Il est obligé de généraliser sa réflexion, devenant ainsi infiniment plus intelligent face au monde réel.

---

## Méthodologie et Expérimentation (Rapport Technique)

Afin d'offrir une complète transparence sur les mécanismes de cette expérience prouvant concrètement l'intérêt des données stochastiques, voici l'anatomie exacte du flux de recherche implémenté dans nos scripts :

### 1. Prétraitement des Données (Data Processing)
* **Dataset source :** Nous chargeons le standard *Fashion MNIST* (70 000 images réparties en 10 classes de vêtements). 
* **Normalisation de la matrice :** Les valeurs des pixels (0-255) subissent une division formelle par 255.0. Cette conversion des matrices en virgule flottante (0 à 1) limite la variance explosive lors de la descente de gradient, uniformisant la vitesse mathématique d'apprentissage.
* **Stratégie de raréfaction :** Nous scindons drastiquement le dataset pour ne confier au système que **1000 images d’entraînement**. Ce sous-échantillonnage extrême est un "piège de recherche volontaire" visant à réduire radicalement le champ d'expérience de l'ordinateur afin de forcer un effondrement par Surapprentissage très rapide.

### 2. Architecture et Création des Modèles
Nous avons bâti à l'aide de Keras/TensorFlow deux Réseaux de Neurones Convolutionnels (CNN) jumeaux. Ils possèdent tous deux des couches de convolution (Extraction des pixels clés via des filtres), de `MaxPooling` (Réduction du bruit) et un final Neuronal Dense.
La divergence s'opère sur la porte d'entrée :
* **Le Modèle A (Classique/Déterministe) :** Reçoit la grille d'images pure. Il s'entraîne sur un monde parfaitement constant dans lequel l'univers graphique ne change jamais.
* **Le Modèle B (Aléatoire/Stochastique) :** Se voit greffer en pré-couche un moteur de probabilités aveugle. À la nanoseconde précise où l'image pénètre dans son espace cognitif, un dé est jeté, impliquant une rotation dynamique, un renversement horizontal et des taux de zooms fluctuants. Le réseau affronte ainsi une instabilité continuelle.

### 3. Protocole d'Entraînement
* Le cycle d'itération a été encodé sur une durée rigide de **25 époques (Epochs)**. Ce mur temporel de 25 passages a été savamment calculé : il permet de laisser à l'ordinateur le temps d'apprendre sur les 10 premières époques, avant d'assister à la divergence d'épuisement passée la 15ème. 
* L'optimiseur utilisé est the classique algorithme rectificateur *Adam*, croisant une fonction de perte en mode Catégorie multiple (`sparse_categorical_crossentropy`).

### 4. Interprétation Fondamentale (La Preuve Scientifique)
À la lecture des courbes générées, l'observation rejoint la théorie développée dans `projet.md` :
* **Le Phénomène d'Overfitting Actif (Modèle A) :** Lors des premières itérations, l'absence de bruit semble rassurer le Modèle, au point qu'il engloutit l'information avec une Loss descendante idéale... jusqu'à ce que la courbe bleue (Jeu inconnu / Test) divorce brutalement des résultats pour grimper en flèche. L'ordinateur classique s'est figé sur les détails de l'arrière-plan de l'image (mémorisation à 100%), détruisant sa capacité de généraliser sa déduction face un élément externe.
* **La Régularisation par le Hasard (Modèle B) :** L'ordinateur stochastique, sans cesse agressé (images tordues, de travers), est dans l'incapacité la plus totale de réaliser ce même travail d'apparence. Son "taux d'erreur" peine légèrement plus au démarrage. Mais cette difficulté induite déploie un miracle mathématique : ses deux courbes (apprentissage puis passage sur des photos Tests inconnues) plongent et se stabilisent harmonieusement et conjointement vers l'optimum global. Paradoxalement, c'est l'insertion mathématique de cette instabilité destructrice qui a garanti à la machine la robustesse immunitaire nécessaire pour réussir.

---

## Structure du Projet
```text
📂 augmentation_stochastique_vision/
 ┣ 📂 data                   # Sérialisation brutale (.csv) du calcul de nos erreurs modèles  
 ┣ 📂 models                 # Checkpoints tensoriels locaux
 ┣ 📂 presentation_assets    # Matrices comparatives (L'Origine face à son clone Stochastique)
 ┣ 📜 01_Regularisation_Vision.py
 ┣ 📜 app.py                 # Écran comparatif (Dashboard pédagogique) en Streamlit
 ┗ 📜 requirements.txt
```

## Guide d'Installation
Afin d'intégrer les mathématiques de "Random Rotation" via réseaux de neurones, la lourde bibliothèque `tensorflow` est employée.
```bash
pip install -r requirements.txt
```

## Guide d'Utilisation
La formation originelle et massive des matrices Fashion MNIST s'est déroulée en amont dans `01_Regularisation_Vision.py`.
Pour examiner sereinement le fruit intellectuel de l'expérience, exposez notre Interface Web Analytique :
```bash
streamlit run app.py
```

### 🧭 Navigation & Expérience sur l'Interface (UI)
La navigation de cet environnement a été construite verticalement pour la démonstration académique :
1. **Phase de Sensibilisation Visuelle (Section Haute) :** Commencez par observer la mosaïque de la ligne "Originale", et comparez-la directement à la version "Modifiée Aléatoirement". Expliquez que cette inclinaison stochastique minime, bien qu'imperceptible pour un humain, représente un monde mathématique tout autre et inconnu pour l'Intelligence Artificielle en face de vous, l'empêchant techniquement de répéter son "par-cœur".
2. **Phase Analytique et Conclusionnelle (Section Basse) :** Cette vaste zone confronte deux graphiques de *Fonction de Perte (Loss)* très clairs pour le jury.
   - Pointez immédiatement la catastrophe du bloc de Gauche (L'intelligence artificielle Déterministe) : la fameuse courbe rouge (L'erreur de Test). Celle-ci ne cesse de "grimper vers le ciel", symbole fondamental et indiscutable qu'elle ne comprend plus rien et a sombré dans l'Overfitting total par mémorisation brute.
   - Accueillez alors du regard le contraste absolu du bloc de Droite (L'intelligence Artificielle Stochastique) : la ligne bleue, assagie et docile, descend harmonieusement de pair avec son apprentissage.
   - **Conclusion :**  Le hasard nous a garanti et mathématiquement protégé la généralisation.
