import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

# --- Configuration de la page ---
st.set_page_config(page_title="Telco Churn Analytics", page_icon="📡", layout="wide")

# --- Chargement des modèles ---
@st.cache_resource
def load_models():
    models = {}
    try:
        models['xgboost'] = joblib.load('models/model_stochastique_xgboost.pkl')
        models['lightgbm'] = joblib.load('models/model_stochastique_lightgbm.pkl')
        models['logreg'] = joblib.load('models/model_deterministe_logreg.pkl')
        models['random_forest'] = joblib.load('models/model_stochastique_random_forest.pkl')
        models['sgd'] = joblib.load('models/model_stochastique_sgd.pkl')
        return models
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles. Détail : {e}")
        st.stop()

models = load_models()

# --- Chargement du dataset originel (pour schémas) ---
@st.cache_data
def get_empty_df():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", nrows=1)
    df = df.drop(columns=['Churn', 'customerID'])
    return df

base_df = get_empty_df()

st.title("📡 Projet Telco Churn : Déterminisme vs Stochastique")
st.markdown("Interface d'exploration permettant d'évaluer concrètement la différence de prédiction entre l'approche déterministe mathématique classique et les algorithmes stochastiques d'ensemble.")

# Séparation de l'app en deux onglets
tab_pred, tab_analytics = st.tabs(["🎯 Prédicteur Interactif Multi-Modèles", "📊 Architecture & Visualisations Analytiques"])

# =========================================================
# ONGLET 1 : PREDICTION
# =========================================================
with tab_pred:
    col_input, col_results = st.columns([1, 1.2])

    with col_input:
        st.subheader("Saisie du Profil Client")
        
        tenure = st.slider("Ancienneté (tenure en mois)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Charges Mensuelles ($)", min_value=15.0, max_value=120.0, value=50.0)
        total_charges = tenure * monthly_charges
        
        contract = st.selectbox("Type de Contrat", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Sécurité en Ligne", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
        tech_support = st.selectbox("Support Technique", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
        payment_method = st.selectbox("Méthode de Paiement", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        predict_btn = st.button("Lancer la Prévision Multi-Modèles", type="primary")

    with col_results:
        st.subheader("Résultats et Comparaison Algorithmique")
        
        if predict_btn:
            user_data = base_df.copy()
            user_data.at[0, 'tenure'] = tenure
            user_data.at[0, 'MonthlyCharges'] = monthly_charges
            user_data.at[0, 'TotalCharges'] = total_charges
            user_data.at[0, 'Contract'] = contract
            user_data.at[0, 'InternetService'] = internet_service
            user_data.at[0, 'OnlineSecurity'] = online_security
            user_data.at[0, 'TechSupport'] = tech_support
            user_data.at[0, 'PaymentMethod'] = payment_method
            
            # Calcul des probabilités pour chaque modèle
            prob_logreg = models['logreg'].predict_proba(user_data)[0][1]
            prob_xgb = models['xgboost'].predict_proba(user_data)[0][1]
            prob_lgb = models['lightgbm'].predict_proba(user_data)[0][1]
            prob_rf = models['random_forest'].predict_proba(user_data)[0][1]
            prob_sgd = models['sgd'].predict_proba(user_data)[0][1]
            
            # Affichage en colonnes
            st.markdown("### Probabilité de Désabonnement (Churn)")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Régression Log.\n(Déterministe)", f"{prob_logreg:.1%}")
            with c2:
                st.metric("Random Forest\n(Stochastique)", f"{prob_rf:.1%}")
            with c3:
                st.metric("XGBoost\n(Stochastique)", f"{prob_xgb:.1%}")
            with c4:
                st.metric("LightGBM\n(Stochastique)", f"{prob_lgb:.1%}")
            with c5:
                st.metric("SGD Classif.\n(Stochastique)", f"{prob_sgd:.1%}")
                
            st.divider()


# =========================================================
# ONGLET 2 : ANALYTIQUE ET VISUELS
# =========================================================
with tab_analytics:
    st.header("Exploration Visuelle et Validation Scientifique")
    
    def display_img(path, caption):
        if os.path.exists(path):
            st.image(Image.open(path), caption=caption, use_container_width=True)
        else:
            st.warning(f"Image {path} introuvable.")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.subheader("Structure et Corrélation")
        display_img("presentation_assets/1_déséquilibre_classes.png", "Axiome : Déséquilibre massif de la variable cible")
        st.markdown("""
        > 💡 **Comment expliquer l'image du Déséquilibre ?**  
        > L'écrasante majorité des clients de ce tableau sont stables (Non), contre très peu de résiliations (Oui). Cela explique pourquoi une approche mathématique classique échoue : elle pourrait tricher, voter bêtement "Non" pour tout le monde, et obtenir une très bonne "note mathématique" (Accuracy) de 73% alors qu'elle n'a strictement rien deviné de notre métier !
        """)
        display_img("presentation_assets/2_correlations_heatmap.png", "Carte de chaleur des covariances")
        st.markdown("""
        > 💡 **Comment expliquer cette Carte de Chaleur (Heatmap) ?**  
        > Elle montre comment nos données se superposent et se "polluent". Les couleurs chaudes signalent des éléments croisés très ressemblants. Contrairement aux approches déterministes qui s'emmêlent les pinceaux devant trop de ressemblance, l'arbre stochastique isole les éléments au hasard sans se laisser biaiser.
        """)
        
    with row1_c2:
        st.subheader("Performance Mathématique et Éthique")
        display_img("presentation_assets/5_comparaison_roc_auc.png", "Duel des trajectoires algorithmiques (Courbes ROC)")
        st.markdown("""
        > 💡 **Comment lire le graphique des Courbes ROC ?**  
        > Dans ce duel, plus une ligne "gonfle" vers le carré en haut à gauche, plus l'ordinateur est souverain et intelligent (Performances de 100%).
        > * **La ligne la plus basse (Régression logistique) :** Modèle déterministe. En tirant une simple règle rigide, il percute un plafond de verre et plafonne tout en bas.
        > * **Toutes les autres courbes supérieures :** Ce sont les modèles probabilitistes (Stochastiques). Grâce au hasard et à l'intelligence de la foule (des milliers de tirages), ils s'adaptent mieux à la complexité de l'humain et dépassent sans effort la machine classique sur notre écran !
        """)
        
        st.subheader("Explicabilité (Interpretability) croisée")
        st.markdown("Examinez l'explication mécanique du 'cerveau' de la machine selon son paradigme :")
        model_choice = st.selectbox("Sélectionnez le modèle à analyser :", 
                                    ["Régression Logistique (Déterministe)", "Random Forest (Stochastique)", 
                                     "XGBoost (Stochastique)", "LightGBM (Stochastique)"])
        
        if model_choice == "Régression Logistique (Déterministe)":
            display_img("presentation_assets/6_importance_logreg.png", "Coefficients Linéaires Absolus")
        elif model_choice == "Random Forest (Stochastique)":
            display_img("presentation_assets/6_importance_rf.png", "Gains d'Information (Random Forest)")
        elif model_choice == "XGBoost (Stochastique)":
            display_img("presentation_assets/6_importance_xgb.png", "Gains d'Information (XGBoost)")
        else:
            display_img("presentation_assets/6_importance_lgb.png", "Gains d'Information (LightGBM)")
            
        st.markdown("""
        > 💡 **Comment lire ces graphiques en Barres (Feature Importance) ?**  
        > Ce comparatif est redoutable car il vous permet de montrer à votre jury "ce qui se passe dans la tête" des IA. Quelles colonnes Excel les intéressent le plus ?
        > * **Si vous êtes sur la Régression (Déterministe) :** L'intelligence est coincée. L'ordinateur affiche des poids (positifs ou négatifs) fixes. La machine considère de la même manière robotique l'Ancienneté d'un client, qu'il ait 20 ans ou 80 ans. Toute subtilité est écrasée.
        > * **Si vous passez sur un modèle Stochastique :** Vous remarquerez que la liste change et s'uniformise ! En divisant l'apprentissage au hasard, l'arbre stochastique accorde beaucoup plus de respect humain : il prend en compte les croisements d'informations subtils ("Le client est jeune MAIS il paie pas cher = Intention différente"). C'est la beauté du modèle ensembliste.
        """)
