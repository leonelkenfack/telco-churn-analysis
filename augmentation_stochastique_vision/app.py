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
st.subheader("1. Le Coupable : 'Le Hasard Artificiel'")
st.markdown("Pour l'ordinateur, un simple hasard de rotation de 5° ou un micro-flou aléatoire modifie l'intégrité absolue de la matrice de pixel. Autrement dit : **le réseau stochastique ne s'entraîne mathématiquement jamais deux fois sur la même image** alors que le réseau déterministe oui.")
st.image(img_bruit, use_container_width=False)

st.divider()

# ANALYSE PUREMENT MATHEMATIQUE
st.subheader("2. La Preuve Scientifique : Courbes des Pertes (Loss)")
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
