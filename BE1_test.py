# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 19:33:14 2025

@author: Julien
"""

# -*- coding: utf-8 -*-
"""
Projet : Qui utilise mon appli ? (Copilote)
But : charger train/test, construire >=10 features manuelles, entraîner plusieurs modèles,
      optimiser F1-macro et produire une submission.csv prête pour Kaggle.
Usage : adapter ds_name puis exécuter.
Python : 3.13 (librairies listées dans requirements.txt)
"""

# === Imports ===
import re
from collections import Counter, defaultdict
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# optional: xgboost (if installed) for stronger model
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# === Config utilisateur ===
ds_name_train = "train"   # si tu veux "train" ou le nom du fichier sans extension
ds_name_test = "test"
base_path = r"D:\Centrale Lyon 20252026\Data science\CSV"
path_template = fr"{base_path}\{{}}.csv"   # .format(ds_name)
TRAIN_PATH = path_template.format(ds_name_train)
TEST_PATH = path_template.format(ds_name_test)

RANDOM_STATE = 42

# === Fonctions utilitaires de lecture (robuste aux lignes inégales) ===
def read_ds(path: str) -> pd.DataFrame:
    """
    Lit le CSV comme dans l'énoncé, construit un DataFrame où chaque colonne suivante est une action.
    Retourne DataFrame avec colonnes : util (si présente), navigateur, 0,1,2... actions
    """
    # On lit ligne par ligne en split virgule (les données semblent déjà séparées par virgule)
    rows = []
    max_cols = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n\r")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            rows.append(parts)
            if len(parts) > max_cols:
                max_cols = len(parts)
    # Construire noms colonnes
    # On détecte si le premier champ est un utilisateur (train) ou pas (test)
    first = rows[0]
    if len(first) >= 2 and first[1] in ["Firefox", "Google Chrome", "Microsoft Edge", "Opera"]:
        # a priori first[0] is util, first[1] browser
        header = ["util", "navigateur"] + [str(i) for i in range(max_cols - 2)]
    else:
        header = ["navigateur"] + [str(i) for i in range(max_cols - 1)]
    # Pad rows
    fixed = []
    for r in rows:
        r2 = r + [np.nan] * (max_cols - len(r))
        fixed.append(r2)
    df = pd.DataFrame(fixed, columns=header)
    return df

# === Charger données ===
print("Lecture des données...")
features_train = read_ds(TRAIN_PATH)
features_test = read_ds(TEST_PATH)
print("Train shape:", features_train.shape, "Test shape:", features_test.shape)

# === Aperçu rapide ===
print("\nQuelques utilisateurs (train):", features_train['util'].unique()[:10])
print("Navigateurs (train):", features_train['navigateur'].value_counts().to_dict())

# === Helpers pour parser actions ===
re_screen = re.compile(r"\((.*?)\)")
re_conf = re.compile(r"<(.*?)>")
re_chain = re.compile(r"\$(.*?)\$")
re_t = re.compile(r"t(\d+)")
def is_time_token(tok: str):
    return isinstance(tok, str) and tok.startswith("t") and tok[1:].isdigit()

def clean_action(tok: str) -> str:
    """Supprime les détails entre () <> $ $ et les suffixes '1', retourne action de base"""
    if not isinstance(tok, str):
        return ""
    # retirer les balises de temps
    if is_time_token(tok):
        return tok  # keep time tokens as-is if needed
    s = tok
    # enlever contenu entre parenthèses (écrans), garder le préfixe action
    if "(" in s:
        s = s.split("(")[0]
    if "<" in s:
        s = s.split("<")[0]
    # enlever trailing $...$
    if "$" in s:
        s = s.split("$")[0]
    # enlever trailing '1' (modification mark)
    if s.endswith("1"):
        s = s[:-1]
    return s.strip()

# === Feature engineering personnalisé (>=10 features) ===
# On va produire :
# 1) total_actions : nombre d'actions effectives (exclut NaN et tokens temps)
# 2) n_time_windows : nombre distinct de tokens tXX présents (approx durée)
# 3) max_time : valeur maximale tXX (en secondes) -> estimation durée session
# 4) unique_actions : nombre d'actions uniques (types)
# 5) top_action_count : compte de l'action la plus fréquente
# 6) ratio_top_action : top_action_count / total_actions
# 7) nb_screens : nombre d'écrans distincts extraits via parenthèses
# 8) nb_conf_screens : nombre de conf écran distinctes (<>)
# 9) nb_chain_refs : nombre d'occurrences de $...$ (fiche)
# 10) nb_modified_actions : nombre d'actions terminant par '1'
# 11) entropy_actions : entropie de la distribution d'actions (mesure comportement)
# 12) browser_OHE (4 colonnes) -> encodées plus tard
# 13) mean_actions_per_timewindow : moyenne actions par fenêtre tXX (proxy d'activité)
# 14) toasts_count : nombre 'Affichage d'un toast' (action indicative)
# 15) errors_count : nombre 'Affichage d'une erreur' (action indicative)

# On crée un transformer custom pour vectoriser les actions textuelles (top-k counts)
class SessionFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, action_cols_prefix=None, top_k_actions=50):
        self.top_k_actions = top_k_actions
        self.action_cols_prefix = action_cols_prefix
        self.top_actions_ = None

    def fit(self, X, y=None):
        # X is DataFrame
        # collect global action frequencies (cleaned)
        cnt = Counter()
        for idx, row in X.iterrows():
            # actions start after 'navigateur' (and maybe 'util')
            # we consider all columns other than 'util' and 'navigateur'
            for tok in row.drop(labels=[c for c in ['util', 'navigateur'] if c in row.index], errors='ignore'):
                if pd.isna(tok):
                    continue
                tok = str(tok)
                if is_time_token(tok):
                    continue
                a = clean_action(tok)
                if a:
                    cnt[a] += 1
        # top k actions
        self.top_actions_ = [a for a, _ in cnt.most_common(self.top_k_actions)]
        return self

    def transform(self, X):
        rows = []
        for idx, row in X.iterrows():
            toks = [tok for tok in row.drop(labels=[c for c in ['util', 'navigateur'] if c in row.index], errors='ignore') if not pd.isna(tok)]
            total_actions = sum(1 for tok in toks if not is_time_token(str(tok)))
            # time windows
            t_vals = [int(re_t.match(tok).group(1)) for tok in toks if isinstance(tok, str) and re_t.match(tok)]
            n_time_windows = len(t_vals)
            max_time = max(t_vals) if t_vals else 0
            # actions cleaned
            cleaned = [clean_action(tok) for tok in toks if not is_time_token(str(tok))]
            unique_actions = len(set(cleaned))
            # counts
            c = Counter(cleaned)
            top_action_count = c.most_common(1)[0][1] if c else 0
            ratio_top = top_action_count / total_actions if total_actions>0 else 0
            # screens & conf
            screens = []
            confs = []
            chain_count = 0
            modified_count = 0
            toasts = 0
            errors = 0
            # actions per time window: map tXX->count
            actions_per_window = defaultdict(int)
            current_window = 0
            for tok in toks:
                s = str(tok)
                mt = re_t.match(s)
                if mt:
                    current_window = int(mt.group(1))
                    continue
                # it's an action
                actions_per_window[current_window] += 1
                if "$" in s:
                    chain_count += s.count("$")//2 + 1  # approx
                if s.endswith("1"):
                    modified_count += 1
                if "Affichage d'un toast" in s:
                    toasts += 1
                if "Affichage d'une erreur" in s:
                    errors += 1
                # screens
                m = re_screen.search(s)
                if m:
                    screens.append(m.group(1))
                m2 = re_conf.search(s)
                if m2:
                    confs.append(m2.group(1))
            nb_screens = len(set(screens))
            nb_confs = len(set(confs))
            mean_actions_per_window = np.mean(list(actions_per_window.values())) if actions_per_window else 0
            # entropy
            probs = np.array(list(c.values()), dtype=float)
            if probs.size > 0:
                probs = probs / probs.sum()
                entropy = -(probs * np.log2(probs + 1e-12)).sum()
            else:
                entropy = 0.0
            # top-k action indicator vector
            topk_counts = [c.get(a, 0) for a in self.top_actions_] if self.top_actions_ else [0]*self.top_k_actions

            row_feat = {
                "total_actions": total_actions,
                "n_time_windows": n_time_windows,
                "max_time": max_time,
                "unique_actions": unique_actions,
                "top_action_count": top_action_count,
                "ratio_top_action": ratio_top,
                "nb_screens": nb_screens,
                "nb_confs": nb_confs,
                "nb_chain_refs": chain_count,
                "nb_modified_actions": modified_count,
                "entropy_actions": entropy,
                "mean_actions_per_window": mean_actions_per_window,
                "toasts_count": toasts,
                "errors_count": errors
            }
            # add topk counts with prefix
            for i, cval in enumerate(topk_counts):
                row_feat[f"topk_{i}"] = cval
            rows.append(row_feat)
        return pd.DataFrame(rows, index=X.index)

# === Construire features pour train + test (même pipeline) ===
print("\nConstruction des features (train)...")
extractor = SessionFeatureExtractor(top_k_actions=50)
extractor.fit(features_train)
train_feats = extractor.transform(features_train)
print("Train feats shape:", train_feats.shape)

print("Construction des features (test)...")
# Pour test, il faut utiliser même top_actions_ (fit sur train)
test_feats = extractor.transform(features_test)
print("Test feats shape:", test_feats.shape)

# === Encodage navigateur et util (label target) ===
# Navigateur -> one-hot
browsers = pd.get_dummies(features_train['navigateur'].fillna("unknown"))
# appliquer même colonnes sur test
browsers_test = pd.get_dummies(features_test['navigateur'].fillna("unknown"))
# align columns
browsers_test = browsers_test.reindex(columns=browsers.columns, fill_value=0)

# Concat features + browser OHE
X_train = pd.concat([train_feats.reset_index(drop=True), browsers.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_feats.reset_index(drop=True), browsers_test.reset_index(drop=True)], axis=1)

# Target encode
le = LabelEncoder()
y = le.fit_transform(features_train['util'].astype(str).values)

print("\nX_train shape after concat:", X_train.shape)

# === Baseline train/validation split (stratified) ===
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print("Split sizes:", X_tr.shape, X_val.shape)

# === Modèles à tester rapidement ===
models = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    ])
}
if HAS_XGBOOST:
    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE))
    ])

# === Fonction d'évaluation F1-macro (CV) ===
def evaluate_models(models_dict, X, y, cv=5):
    results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models_dict.items():
        print(f"\nEvaluation CV pour {name} ...")
        scores = cross_validate(model, X, y, scoring="f1_macro", cv=skf, n_jobs=-1, return_train_score=True)
        results[name] = {
            "train_f1": scores["train_score"].mean(),
            "val_f1": scores["test_score"].mean(),
            "std": scores["test_score"].std()
        }
        print(f"{name} -- train f1: {results[name]['train_f1']:.4f}, cv f1: {results[name]['val_f1']:.4f} (+/- {results[name]['std']:.4f})")
    return results

# Evaluate quickly
cv_results = evaluate_models(models, X_tr, y_tr, cv=4)

# === Entraînement final sur tout train + prédiction sur validation ===
best_name = max(cv_results.items(), key=lambda x: x[1]['val_f1'])[0]
print("\nMeilleur modèle choisi (CV):", best_name)
best_model = models[best_name]
best_model.fit(X_tr, y_tr)
y_val_pred = best_model.predict(X_val)
f1_val = f1_score(y_val, y_val_pred, average="macro")
print(f"F1-macro sur set de validation: {f1_val:.4f}")
print("\nClassification report (validation):\n", classification_report(y_val, y_val_pred, target_names=le.classes_))

# === Feature importance (si RandomForest) ===
if best_name == "RandomForest":
    rf = best_model.named_steps["clf"]
    # récupérer noms features
    feat_names = X_train.columns.tolist()
    importances = rf.feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)
    print("\nTop 30 features (RF):")
    print(fi)
    # plot
    plt.figure(figsize=(8,6))
    fi.plot.barh()
    plt.gca().invert_yaxis()
    plt.title("Top 30 feature importances (RandomForest)")
    plt.tight_layout()
    plt.show()

# === Matrice de confusion (optionnelle, sur validation) ===
# On prend les 15 classes les plus fréquentes pour la matrice afin d'être lisible
vc = Counter(y_val)
most_common_classes = [c for c, _ in vc.most_common(15)]
mask = np.isin(y_val, most_common_classes)
if mask.sum() > 0:
    cm = confusion_matrix(y_val[mask], y_val_pred[mask], labels=most_common_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(most_common_classes))
    fig, ax = plt.subplots(figsize=(10,8))
    disp.plot(ax=ax, xticks_rotation=90, cmap=plt.cm.Blues)
    plt.title("Confusion matrix (top-15 classes in validation)")
    plt.show()

# === Entraînement final sur tout TRAIN (pour submission) ===
print("\nEntraînement final sur l'ensemble train complet...")
best_model.fit(X_train, y)

# Préparer X_test (déjà fait), appliquer imputer via pipeline when predicting
preds_test_codes = best_model.predict(X_test)
preds_test_labels = le.inverse_transform(preds_test_codes)

# Construire DataFrame de submission selon consignes (RowId start 1)
df_subm = pd.DataFrame({"prediction": preds_test_labels})
df_subm.index = df_subm.index + 1
df_subm.index.name = "RowId"
print("\nAperçu soumission:")
print(df_subm.head())

out_path = os.path.join(base_path, "submission.csv")
df_subm.to_csv(out_path)
print(f"\nSubmission saved to: {out_path}")

# === Graphiques utiles pour le rapport ===
# 1) distribution des classes (train)
plt.figure(figsize=(10,4))
pd.Series(y).value_counts().sort_values(ascending=False).head(40).plot.bar()
plt.title("Distribution (counts) des 40 classes les plus fréquentes - train")
plt.xlabel("class (encoded)")
plt.ylabel("count")
plt.tight_layout()
plt.show()

# 2) relation entre total_actions et max_time
plt.figure(figsize=(8,5))
plt.scatter(X_train['max_time'].clip(0,1000), X_train['total_actions'], alpha=0.3, s=6)
plt.xlabel("max_time (s) clipped at 1000")
plt.ylabel("total_actions")
plt.title("Max_time vs total_actions (train)")
plt.tight_layout()
plt.show()

# === Fin ===
print("\nTerminé. Résumé:")
for k, v in cv_results.items():
    print(f"{k} : CV f1={v['val_f1']:.4f} (std {v['std']:.4f})")
print(f"Validation F1 (final split) pour {best_name}: {f1_val:.4f}")
print("Submission écrite :", out_path)
