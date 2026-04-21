import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, recall_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Importation pour LightGBM et XGBoost
import lightgbm as lgb
import xgboost as xgb

sns.set_theme(style="whitegrid")

print("Chargement des données prétraitées...")
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze('columns') # Convert to Series
y_test = pd.read_csv("y_test.csv").squeeze('columns') # Convert to Series

# Identification des types de colonnes
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
# Exclusion de l'ID qui ne sert à rien
if 'customerID' in categorical_cols:
    categorical_cols.remove('customerID')
    X_train = X_train.drop(columns=['customerID'])
    X_test = X_test.drop(columns=['customerID'])

numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Construction du Pipeline de prétraitement...")
# Pipeline pour normaliser les variables continues et encoder les catégorielles
# Le ColumnTransformer applique OneHotEncoder en évitant le piège des variables corrélées grâce à drop='first' ou handle_unknown='ignore'
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fonction d'évaluation (pas l'accuracy)
def evaluate_model(model_name, y_true, y_pred, y_prob):
    print(f"\n--- Résultats pour {model_name} ---")
    print("F1-Score:", f1_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    # Affichage Matrice de confusion modeste (Optionnel à l'écran)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    return roc_curve(y_true, y_prob)

models_roc = {}

# 1. Modèle Déterministe de Référence : Régression Logistique
print("\n[ ENTRAINEMENT ] Phase 1 : Régression Logistique (Déterministe)")
pipe_lr = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = evaluate_model("Régression Logistique", y_test, y_pred_lr, y_prob_lr)
models_roc["LogReg"] = (fpr_lr, tpr_lr, roc_auc_score(y_test, y_prob_lr))

# 2. Modèle Stochastique n°1 : Random Forest
print("\n[ ENTRAINEMENT ] Phase 2 : Random Forest (Bagging Stochastique)")
pipe_rf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])
# Pour accélérer on entraine directement sans HP tuning massif, on cherche jute à montrer le comportement stochastique
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)
y_prob_rf = pipe_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)
models_roc["Random Forest"] = (fpr_rf, tpr_rf, roc_auc_score(y_test, y_prob_rf))

# 3. Modèle Stochastique n°2 : XGBoost
print("\n[ ENTRAINEMENT ] Phase 3 : XGBoost (Stochastic Gradient Boosting)")
pipe_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
pipe_xgb.fit(X_train, y_train)
y_pred_xgb = pipe_xgb.predict(X_test)
y_prob_xgb = pipe_xgb.predict_proba(X_test)[:, 1]
fpr_xgb, tpr_xgb, _ = evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)
models_roc["XGBoost"] = (fpr_xgb, tpr_xgb, roc_auc_score(y_test, y_prob_xgb))

# 4. Modèle Stochastique n°3 : LightGBM
print("\n[ ENTRAINEMENT ] Phase 4 : LightGBM")
pipe_lgb = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))])
pipe_lgb.fit(X_train, y_train)
y_pred_lgb = pipe_lgb.predict(X_test)
y_prob_lgb = pipe_lgb.predict_proba(X_test)[:, 1]
fpr_lgb, tpr_lgb, _ = evaluate_model("LightGBM", y_test, y_pred_lgb, y_prob_lgb)
models_roc["LightGBM"] = (fpr_lgb, tpr_lgb, roc_auc_score(y_test, y_prob_lgb))

# 5. Modèle Stochastique n°4 : SGD Classifier
print("\n[ ENTRAINEMENT ] Phase 5 : SGD Classifier (Stochastic Gradient Descent)")
# SGDClassifier utilise un loss 'log' / 'log_loss' pour faire de la régression logistique via descente stochastique
pipe_sgd = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', SGDClassifier(loss='log_loss', random_state=42, max_iter=1000))])
pipe_sgd.fit(X_train, y_train)
y_pred_sgd = pipe_sgd.predict(X_test)
y_prob_sgd = pipe_sgd.predict_proba(X_test)[:, 1]
fpr_sgd, tpr_sgd, _ = evaluate_model("SGD Classifier", y_test, y_pred_sgd, y_prob_sgd)
models_roc["SGD Classifier"] = (fpr_sgd, tpr_sgd, roc_auc_score(y_test, y_prob_sgd))

# ==== EXPORT VISUELS ====
# Comparaison ROC-AUC
plt.figure(figsize=(10, 8))
for model_name, (fpr, tpr, auc_score) in models_roc.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Comparaison des Courbes ROC (Déterministe vs Stochastique)')
plt.legend(loc='lower right')
plt.savefig("presentation_assets/5_comparaison_roc_auc.png", bbox_inches='tight')
plt.close()

# ================= MULTI-MODEL FEATURE IMPORTANCE =================
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(cat_feature_names)

def plot_feature_importance(importances, title, filename, is_coef=False):
    if is_coef:
        # coefficients linéaires, on trie par valeur absolue
        df_imp = pd.DataFrame({'Feature': all_feature_names, 'Coef': importances})
        df_imp['Abs_Coef'] = df_imp['Coef'].abs()
        df_imp = df_imp.sort_values(by='Abs_Coef', ascending=False).head(15)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Coef', y='Feature', data=df_imp, palette='coolwarm', hue='Feature', legend=False)
        plt.title(title)
    else:
        df_imp = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis', hue='Feature', legend=False)
        plt.title(title)
        
    plt.savefig(f"presentation_assets/{filename}", bbox_inches='tight')
    plt.close()

# 1. LogReg
lr_model = pipe_lr.named_steps['classifier']
plot_feature_importance(lr_model.coef_[0], "Coefficients Linéaires (Régression Logistique)", "6_importance_logreg.png", is_coef=True)

# 2. Random Forest
rf_model = pipe_rf.named_steps['classifier']
plot_feature_importance(rf_model.feature_importances_, "Gains d'Information (Random Forest)", "6_importance_rf.png")

# 3. XGBoost
xgb_model = pipe_xgb.named_steps['classifier']
plot_feature_importance(xgb_model.feature_importances_, "Gains d'Information (XGBoost)", "6_importance_xgb.png")

# 4. LightGBM
lgb_model = pipe_lgb.named_steps['classifier']
plot_feature_importance(lgb_model.feature_importances_, "Gains d'Information (LightGBM)", "6_importance_lgb.png")


# Sauvegarde de l'architecture du pipeline pour illustration
# En python natif, la simple représentation sert de preuve de la rigueur
with open("presentation_assets/3_pipeline_architecture.txt", "w") as f:
    f.write(str(preprocessor))

print("\nSauvegarde des modèles pour Streamlit")
joblib.dump(pipe_lr, 'model_deterministe_logreg.pkl')
joblib.dump(pipe_rf, 'model_stochastique_random_forest.pkl')
joblib.dump(pipe_xgb, 'model_stochastique_xgboost.pkl')
joblib.dump(pipe_lgb, 'model_stochastique_lightgbm.pkl')
joblib.dump(pipe_sgd, 'model_stochastique_sgd.pkl')

print("Exécution complète et visuels exportés dans presentation_assets/")
