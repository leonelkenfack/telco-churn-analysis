# Recherche : Régularisation d'Images par le Hasard Temporel

## Description
Cette expérimentation finale porte sur l'Intelligence Artificielle de reconnaissance d'images. Notre but est de rendre un concept informatique complexe totalement accessible : **comprendre pourquoi une Intelligence Artificielle "apprend mal" si on ne l'entraîne pas avec un peu de hasard.**

Comme abordé dans notre rapport de recherche au niveau du **Chapitre 3.2 (Stratégies d'Augmentation Stochastique de Données)**, une machine trop rigide (déterministe) a un grand défaut : elle fait du "par cœur" (phénomène appelé surapprentissage). Si on lui donne à apprendre 1000 photos parfaites de vêtements, elle va simplement les mémoriser comme un robot, mais paniquera totalement si on lui donne demain une vraie photo d'un client légèrement de travers ou floue. 

Pour protéger l'Intelligence Artificielle de cette mémorisation "bête", ce scénario démontre sur nos écrans l'approche Stochastique. L'idée est simple : on va volontairement demander au hasard de "brouiller" les images. À chaque fois qu'on montre un vêtement à l'Intelligence Artificielle lors de sa formation, une probabilité va incliner la photo, zoomer ou la retourner l'image (effet miroir). 
L'ordinateur ne voyant techniquement plus la même image pure : cela l'empêche de tricher et de mémoriser. Il est obligé de généraliser sa réflexion, devenant ainsi infiniment plus intelligent face au monde réel.

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
