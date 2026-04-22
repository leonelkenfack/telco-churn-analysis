import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import urllib.request
import os
import plotly.express as px
import time

st.set_page_config(page_title="Systèmes Multi-Agents (SMA)", page_icon="🦠", layout="wide")

st.title("🦠 Émergence par le Hasard : Simulation Stochastique")
st.markdown("Ce tableau de bord confirme la redoutable puissance des Systèmes Multi-Agents (**Chapitre 3.3**). "
            "Il démontre comment des **probabilités purement locales** jetées de manière chaotique pour chaque individu tracent finalement "
            "la célèbre macro-courbe prédictible de contagion. <br>*(Topology: Graph Social réel Facebook SNAP, 4039 nœuds)*", unsafe_allow_html=True)

@st.cache_resource
def load_graph():
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    file_name = "data/facebook_combined.txt.gz"
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)
    return nx.read_edgelist(file_name, nodetype=int)

with st.spinner("Chargement de la topologie du réseau académique (Stanford)..."):
    G = load_graph()

col_control, col_graph = st.columns([1, 2.5])

with col_control:
    st.subheader("Les Lois Quantiques")
    st.markdown("Forcez l'ordinateur à lancer les dés virtuels :")
    beta = st.slider("Probabilité d'infection par lien (β)", 0.01, 0.40, 0.04, format="%.2f")
    gamma = st.slider("Probabilité de guérison/rejet (γ)", 0.01, 0.20, 0.02, format="%.2f")
    duree = st.slider("Horloge temporelle (jours/itérations)", 20, 150, 80)
    patients_zeros = st.number_input("Nombre de Cas Primaires", 1, 100, 5)
    
    start_sim = st.button("🔴 Déclencher le Système Global", type="primary", use_container_width=True)
    
    st.info("**Paradoxe Scientifique :** \nAucun réseau de neurones n'est utilisé ! L'algorithme ne sait pas dessiner une 'courbe en cloche'. "
            "Il détermine juste mathématiquement à chaque instant *t* si tel monsieur contamine son collègue sur la base d'un Random().")

with col_graph:
    if start_sim:
        st.subheader("Propagation de l'Onde Stochastique en Direct")
        chart_placeholder = st.empty()
        
        estados = {node: 0 for node in G.nodes()} 
        # Initialisation aléatoire
        initial_infected = np.random.choice(G.nodes(), size=patients_zeros, replace=False)
        for p in initial_infected:
            estados[p] = 1
            
        history = []
        progress_bar = st.progress(0)
        
        for step in range(duree):
            new_estados = estados.copy()
            num_s = num_i = num_r = 0
            
            for node in G.nodes():
                e = estados[node]
                if e == 0:
                    num_s += 1
                    infected_neighbors = sum(1 for n in G.neighbors(node) if estados[n] == 1)
                    if infected_neighbors > 0:
                        prob_infection = 1 - (1 - beta)**infected_neighbors
                        if np.random.rand() < prob_infection:
                            new_estados[node] = 1
                elif e == 1:
                    num_i += 1
                    if np.random.rand() < gamma:
                        new_estados[node] = 2
                else:
                    num_r += 1
            
            history.append((step, num_s, num_i, num_r))
            estados = new_estados
            
            # Animation fluide (on met à jour l'UI tous les 2 itérations)
            if step % 2 == 0 or step == duree - 1:
                df_history = pd.DataFrame(history, columns=["Temps", "Vecteurs Sains (S)", "Contaminés (I)", "Immunisés (R)"])
                fig = px.line(df_history, x="Temps", y=["Vecteurs Sains (S)", "Contaminés (I)", "Immunisés (R)"], 
                              color_discrete_map={"Vecteurs Sains (S)": "#1f77b4", "Contaminés (I)": "#d62728", "Immunisés (R)": "#2ca02c"},
                              title=f"Itération temporelle t={step}/{duree}")
                fig.update_layout(yaxis_title="Volume de la Population du Graphe", legend_title="États Isolés", height=500)
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
            progress_bar.progress((step + 1) / duree)
            
        st.success("✔️ La courbe macroscopique, visiblement lisse, vient formellement d'émerger du pur chaos fractal de 4000 agents individuels. Loi des grands nombres validée.")
    else:
        st.info("👈 Paramétrez (β) et (γ), puis cliquez sur **Déclencher le Système Global** pour lancer les intéractions.")
