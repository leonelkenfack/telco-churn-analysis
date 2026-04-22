import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="Régularisation par l'Aléa", page_icon="👁️", layout="wide")

st.title("👁️ Stochastique Visuelle (Deep Learning)")
st.markdown("Nous abordons ici le **Chapitre 3.2 de la thèse : L'Augmentation Stochastique de données**. "
            "Le principe démontré est qu'une machine sans aléa apprend bêtement par cœur, tandis que projeter une image sous un biais stochastique "
            "l'empêche de sombrer dans l'Overfitting (Surapprentissage). Dataset de Contrôle : **Fashion MNIST (Vêtements)**.")

try:
    df = pd.read_csv("data/historique_vision.csv")
    img_bruit = Image.open("presentation_assets/vision_metrics.png")
except:
    st.error("Les données ne sont pas prêtes. Veuillez exécuter `python 01_Regularisation_Vision.py` ou le Notebook associé.")
    st.stop()

# VISUEL DE LA PERTURBATION
st.subheader("1. La Technique : Brouiller volontairement la vue")
st.markdown("""
> 💡 *Attention à ne pas s'y tromper : L'ordinateur ne **génère** pas de "nouveaux" vêtements de toutes pièces sur la deuxième ligne !*

Les images de la ligne du bas (La version Stochastique) sont en réalité **exactement les mêmes images originales** que celles du haut, mais que notre système a volontairement décidé de "brouiller" lors de la copie. Avant de la regarder, le système jette un dé au hasard (probabilités) et décide de tourner la photo de 15 degrés, de zoomer un peu dessus, ou de la retourner comme dans un miroir (Data Augmentation). 

**Pourquoi faire cela ?** 
Pour une machine classique (Déterministe), une image est une grille figée de milliers de pixels. Si elle voit cette grille 100 fois à l'identique, la machine va juste s'en souvenir par cœur, bêtement. C'est le fléau du surapprentissage (*Overfitting*).
Mais pour l'entraîneur **Stochastique**, l'image change légèrement à chaque fois. Comme la machine ne voit littéralement *jamais deux fois la même grille de pixel parfaite*, elle ne peut rien mémoriser bêtement. Elle se retrouve forcée de vraiment utiliser de l'intelligence pour comprendre l'essence globale du vêtement !
""")
st.image(img_bruit, use_container_width=False)

st.divider()

# ANALYSE DE L'EFFONDREMENT
st.subheader("2. La Preuve Scientifique : Lire l'effondrement et la victoire")
st.markdown("""
> 💡 **Comment expliquer cette courbe à un non-informaticien ?**  
> L'axe vertical de gauche représente "l'Erreur" commise par la machine. Plus c'est bas, mieux c'est.

* 🔴 **Les Courbes Rouges (La Machine Classique/Déterministe) :**  
  Regardez la ligne rouge en pointillés qui plonge : la machine classique s'entraîne par cœur et pense être invincible. Mais regardez sa jumelle, **la ligne Rouge pleine (Test sur images inconnues)** : arrivée au milieu du graphe, l'erreur s'envole littéralement vers le plafond ! C'est la panique à bord (le *surapprentissage*), la machine récitait par cœur et se trouve incapable de reconnaître les vêtements réels.

* 🔵 **Les Courbes Bleues (La Machine Stochastique) :**  
  Observez la majestueuse ligne Bleue pleine. Puisque l'on a sans arrêt *brouillé* en direct sa petite salle d'entraînement, la machine probabiliste ne baisse pas les bras en conditions difficiles ! Sa ligne pleine épouse et suit très sagement sa ligne pointillée vers le bas de l'écran de manière harmonique.

**Bilan visuel :** Même sans mathématiques, **la courbe Rouge classique divorce et explose**, là où **la courbe Bleue stochastique reste sage et prédictible**. Le hasard l'a sauvé !
""")

st.markdown("Voici l'évolution historique de nos deux Réseaux de Neurones Convolutionnels isolés pendant 25 époques (*Le modèle cherche à obtenir la Loss la plus proche de 0 possible*).")

col1, col2 = st.columns(2)

with col1:
    fig_det = px.line(df, x="Epoque", y=["Loss_Train_Det", "Loss_Test_Det"], 
                      title="❌ Approche Déterministe (Le Piège)", 
                      color_discrete_sequence=["green", "red"])
    fig_det.update_layout(yaxis_title="Erreur (Loss)", legend_title="Phase d'Apprentissage")
    st.plotly_chart(fig_det, use_container_width=True)
    st.error("Dramatique ! Le modèle pur a appris ses données de Train par cœur (ligne verte qui s'effondre), mais il est devenu incapable de reconnaître des vêtements de Test (la ligne rouge explose vers le haut). C'est le **surapprentissage total**.")

with col2:
    fig_sto = px.line(df, x="Epoque", y=["Loss_Train_Sto", "Loss_Test_Sto"], 
                      title="✅ Approche Stochastique (Le Salut)", 
                      color_discrete_sequence=["green", "blue"])
    fig_sto.update_layout(yaxis_title="Erreur (Loss)", legend_title="Phase d'Apprentissage")
    st.plotly_chart(fig_sto, use_container_width=True)
    st.success("Protecteur ! L'injection de bruit permanent l'a empêché de stagner. La ligne Test (bleue) continue de descendre gentiment de pair avec l'entraînement. La méthode probabiliste assure une vraie généralisation !")
