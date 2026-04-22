import networkx as nx
import numpy as np
import pandas as pd
import urllib.request
import gzip
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("--- INITIALISATION DU SYSTEME MULTI-AGENTS ---")

url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
file_name = "data/facebook_combined.txt.gz"

if not os.path.exists(file_name):
    print("Telechargement du dataset Stanford Facebook en cours...")
    urllib.request.urlretrieve(url, file_name)
    print("Telechargement termine.")

print("\n1. Construction du Graphe Social...")
G = nx.read_edgelist(file_name, nodetype=int)
nb_noeuds = G.number_of_nodes()
nb_aretes = G.number_of_edges()
print(f"Réseau généré : {nb_noeuds} individus (Nœuds), {nb_aretes} connexions (Arêtes).")

print("\n2. Initialisation des règles Stochastiques (Modèle Compartimental SIR)...")
beta = 0.05  # Probabilité d'infection par interaction (par voisin)
gamma = 0.02 # Probabilité de guérison (passage en mode 'Immunisé / Rétabli')
jours_simulation = 60

# 0 = S (Susceptible / Sain)
# 1 = I (Infecté / Actif)
# 2 = R (Rétabli / Immunisé)
states = {node: 0 for node in G.nodes()}

# Déclaration aléatoire des "Patients Zéros"
np.random.seed(42)
patients_zeros = np.random.choice(G.nodes(), size=10, replace=False)
for p in patients_zeros:
    states[p] = 1

history = []

print(f"\n3. Lancement de la Simulation sur {jours_simulation} itérations...")
for step in range(jours_simulation):
    new_states = states.copy()
    num_s = num_i = num_r = 0
    
    for node in G.nodes():
        etat_actuel = states[node]
        if etat_actuel == 0:
            num_s += 1
            # L'individu est sain. Va-t-il se faire infecter stochastiquement ?
            neighbors = list(G.neighbors(node))
            infected_neighbors = sum(1 for n in neighbors if states[n] == 1)
            if infected_neighbors > 0:
                prob_infection = 1 - (1 - beta)**infected_neighbors
                if np.random.rand() < prob_infection:
                    new_states[node] = 1
        elif etat_actuel == 1:
            num_i += 1
            # L'individu est infecté. Va-t-il guérir ?
            if np.random.rand() < gamma:
                new_states[node] = 2
        else:
            num_r += 1
            
    history.append((step, num_s, num_i, num_r))
    states = new_states

# 4. Exportation et création des preuves
print("\n4. Génération des Actifs Académiques...")
df_history = pd.DataFrame(history, columns=["Jour", "Sains (S)", "Infectés (I)", "Rétablis (R)"])
df_history.to_csv("data/historique_sir.csv", index=False)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_history.set_index("Jour"))
plt.title("Propagation Émergente : Modèle SIR Multi-Agents", fontsize=14)
plt.ylabel("Nombre d'Individus")
plt.xlabel("Temps (Itérations)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("presentation_assets/1_courbe_sir.png", dpi=300)
plt.close()

print("\n[SUCCES] Simulation terminée. Les preuves sont dans presentation_assets/")
