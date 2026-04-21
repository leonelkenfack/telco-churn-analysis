import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# Paramétrage de base pour Seaborn
sns.set_theme(style="whitegrid")
# Création du dossier pour les assets s'il n'existe pas
os.makedirs("presentation_assets", exist_ok=True)

print("Chargement des données...")
# 1. Chargement des données
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Nettoyage de TotalCharges
print("Nettoyage...")
# Remplace les espaces vides par 0 puis convertit en numerique
df['TotalCharges'] = df['TotalCharges'].replace(" ", 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

print("Création du diagramme circulaire...")
plt.figure(figsize=(6, 6))
churn_counts = df['Churn'].value_counts()
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90, explode=[0, 0.1])
plt.title("Distribution de la variable cible : Churn")
plt.savefig("presentation_assets/1_déséquilibre_classes.png", bbox_inches='tight')
plt.close()

# 4. Heatmap des corrélations (variables continues vs cible)
df_corr = df.copy()
df_corr['Churn_num'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
df_corr['SeniorCitizen'] = df_corr['SeniorCitizen'].astype(int)

cols_num = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_num']
corr_matrix = df_corr[cols_num].corr()

print("Création de la heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation de Pearson")
plt.savefig("presentation_assets/2_correlations_heatmap.png", bbox_inches='tight')
plt.close()

# 5. Séparation Train / Test
print("Séparation Train/Test...")
X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes': 1, 'No': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Taille du jeu d'entraînement : {X_train.shape[0]}")
print(f"Taille du jeu de test : {X_test.shape[0]}")

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("EDA et Prétraitement initial terminés.")
