import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

print("--- INITIALISATION DU BANC D'ESSAI COMPLET (Multi-Modèles) ---")

# 1. Chargement et prétraitement basique
df = pd.read_csv("../deterministe_vs_stochastique/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

X_transformed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)
print(f"[OK] Données prêtes : Train ({len(X_train)}), Test ({len(X_test)})")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ==========================================
# DEFINITION DU REGISTRE DES MODELES
# ==========================================
models_config = {
    "Régression Logistique": {
        "model": LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000),
        "param_grid": {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
        "optuna_func": lambda trial: {
            'C': trial.suggest_float('C', 0.01, 10, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
        },
        "allow_grid": True,
        "type": "Déterministe (Linéaire)"
    },
    "SGD Classifier": {
        "model": SGDClassifier(random_state=42, loss='log_loss', n_jobs=-1),
        "param_grid": {'alpha': [1e-4, 1e-3, 1e-2], 'penalty': ['l2', 'l1']},
        "optuna_func": lambda trial: {
            'alpha': trial.suggest_float('alpha', 1e-4, 1e-2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1'])
        },
        "allow_grid": True,
        "type": "Stochastique (Descente)"
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "param_grid": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
        "optuna_func": lambda trial: {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.1]),
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7])
        },
        "allow_grid": True,
        "type": "Stochastique (Arbres)"
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
        "optuna_func": lambda trial: {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
            'max_depth': trial.suggest_categorical('max_depth', [5, 10]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5])
        },
        "allow_grid": False, # Ignoré en Grid pour contrainte de temps (cas école)
        "type": "Stochastique (Arbres)"
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
        "param_grid": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
        "optuna_func": lambda trial: {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.1]),
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7])
        },
        "allow_grid": False, # Ignoré en Grid
        "type": "Stochastique (Arbres)"
    }
}

results = []

for model_name, config in models_config.items():
    print(f"\n============================\n[TRAITEMENT] Modèle : {model_name}\n============================")
    
    # --- 1. RANDOM SEARCH ---
    print("--> Random Search (n_iter=10)...")
    random_search = RandomizedSearchCV(estimator=config["model"], param_distributions=config["param_grid"], 
                                     n_iter=10, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42)
    start_time = time.time()
    random_search.fit(X_train, y_train)
    time_random = time.time() - start_time
    preds_random = random_search.predict_proba(X_test)[:, 1] if hasattr(random_search, "predict_proba") else None
    auc_random = roc_auc_score(y_test, preds_random) if preds_random is not None else 0
    results.append({"Modèle": model_name, "Méthode d'Optimisation": "Random Search", "Temps (s)": time_random, "AUC Test": auc_random, "Famille": config["type"]})
    print(f"   [OK] Random: {time_random:.2f}s | AUC: {auc_random:.4f}")

    # --- 2. OPTUNA (Bayesian) ---
    print("--> Optuna (n_trials=10)...")
    def objective(trial):
        params = config["optuna_func"](trial)
        params['random_state'] = 42
        if 'n_jobs' in config["model"].get_params(): params['n_jobs'] = -1
        model = config["model"].__class__(**params)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    start_time = time.time()
    study.optimize(objective, n_trials=10)
    time_optuna = time.time() - start_time
    
    best_params = study.best_params
    best_params['random_state'] = 42
    if 'n_jobs' in config["model"].get_params(): best_params['n_jobs'] = -1
    best_optuna_model = config["model"].__class__(**best_params)
    best_optuna_model.fit(X_train, y_train)
    
    preds_optuna = best_optuna_model.predict_proba(X_test)[:, 1] if hasattr(best_optuna_model, "predict_proba") else None
    auc_optuna = roc_auc_score(y_test, preds_optuna) if preds_optuna is not None else 0
    results.append({"Modèle": model_name, "Méthode d'Optimisation": "Optuna (Bayésien)", "Temps (s)": time_optuna, "AUC Test": auc_optuna, "Famille": config["type"]})
    print(f"   [OK] Optuna: {time_optuna:.2f}s | AUC: {auc_optuna:.4f}")

    # --- 3. GRID SEARCH (Conditionnel) ---
    if config["allow_grid"]:
        print("--> Grid Search Exhaustif...")
        grid_search = GridSearchCV(estimator=config["model"], param_grid=config["param_grid"], cv=cv, scoring='roc_auc', n_jobs=-1)
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        time_grid = time.time() - start_time
        preds_grid = grid_search.predict_proba(X_test)[:, 1] if hasattr(grid_search, "predict_proba") else None
        auc_grid = roc_auc_score(y_test, preds_grid) if preds_grid is not None else 0
        results.append({"Modèle": model_name, "Méthode d'Optimisation": "Grid Search", "Temps (s)": time_grid, "AUC Test": auc_grid, "Famille": config["type"]})
        print(f"   [OK] Grid: {time_grid:.2f}s | AUC: {auc_grid:.4f}")
    else:
        print("--> Grid Search ignoré (Trop lourd pour un cas école)")
        # On peut simuler que s'il était fait, cela prendrait bcp plus de temps
        pass

# ==========================================
# SYNTHESE VISUELLE ET EXPORT
# ==========================================
df_results = pd.DataFrame(results)

plt.figure(figsize=(14, 8))
sns.scatterplot(
    data=df_results, 
    x='Temps (s)', y='AUC Test', 
    hue='Modèle', style="Méthode d'Optimisation",
    s=300, palette='Set1', alpha=0.9
)

# On dessine des flèches pour relier le Random et l'Optuna d'un même modèle (Optionnel mais joli)
plt.title("Efficacité Algorithmique Multi-Modèles : Temps de Calcul vs Performance (AUC)", fontsize=16)
plt.xlabel("Temps de calcul CPU (Secondes)", fontsize=12)
plt.ylabel("Performance Globale Test (AUC ROC)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("presentation_assets/1_comparaison_temps_score.png", dpi=300)
plt.close()

# Sauvegarde des historiques pour Streamlit
df_results.to_csv("data/resultats_comparaison.csv", index=False)

print("\n[SUCCES] Exécution complète terminée ! Graphique et CSV exportés.")
