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
        st.subheader("Performance vs Durée de recherche")
        st.image("presentation_assets/1_comparaison_temps_score.png", use_container_width=True, caption="Analyse croisée des algorithmes et stratégies")
        st.markdown("""
        > 💡 **Que lire sur ce "Banc d'Essai" (Graphique à bulles) ?**  
        > L'axe horizontal, c'est le Temps. L'axe vertical, c'est l'Intelligence (Performance).  
        > Idéalement, sur un graphique, on veut être en haut (très intelligent) et à gauche (très rapide).  
        > * Regardez la grosse bulle du **Grid Search (Déterministe)** : Elle est coincée tout à droite, ce qui signifie qu'elle est extrêmement lente. Parce qu'elle essaie tout de force.  
        > * Regardez les **méthodes Stochastiques (Bayésiennes comme Optuna)** : Leurs bulles sont collées complètement à gauche ! Le hasard guidé a permis à l'IA de trouver ses réglages presque instantanément tout en visant plus haut en performance.
        """)
    except:
        pass
