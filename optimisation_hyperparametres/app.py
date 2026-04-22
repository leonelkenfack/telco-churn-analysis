import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Stratégies de Recherche Expansives", page_icon="⚙️", layout="wide")

st.title("⚙️ Banc d'essai Complet : Efficacité Stochastique")
st.markdown("Ce tableau de bord valide mathématiquement la Section 3.1. Il oppose la méthode déterministe (`GridSearchCV`) aux approches probabilistes sur l'ensemble du périmètre algorithmique (Cas école optimisé temporellement).")

# 1. Chargement des résultats
try:
    df_results = pd.read_csv("data/resultats_comparaison.csv")
except Exception as e:
    st.warning("Génération des résultats en cours. Veuillez patienter ou exécutez le script d'optimisation...")
    st.stop()

# 2. Table des scores et temps
st.subheader("Bilan CPU et Discrimination AUC")
st.markdown("Démonstration multi-modèles de l'avantage de **l'Optuna (Bayésien)** face aux comportements purement aléatoires ou strictement exhaustifs. (Grid Search désactivé sur les forêts d'arbres massives pour éviter la congestion).")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.dataframe(df_results.style.format({"Temps (s)": "{:.2f}", "AUC Test": "{:.4f}"}), use_container_width=True)
    
with col2:
    try:
        st.image("presentation_assets/1_comparaison_temps_score.png", use_container_width=True, caption="Analyse croisée des algorithmes et stratégies")
    except:
        pass
