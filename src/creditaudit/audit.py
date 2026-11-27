"""Fit models, choose a threshold on validation data, and report group metrics on test data."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET, age_group

BASE = ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE"] + [f"PAY_{i}" for i in ("0", "2", "3", "4", "5", "6")]
BASE += [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
MODELS = {
    "logreg_without_sex": ("logreg", BASE),
    "logreg_with_sex": ("logreg", BASE + ["SEX"]),
    "hgb_without_sex": ("hgb", BASE),
    "hgb_with_sex": ("hgb", BASE + ["SEX"]),
}


def make_estimator(kind, seed):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, random_state=seed)


def best_f1_threshold(y, p):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.8, 76):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        f = 2 * tp / max(pred.sum() + (y == 1).sum(), 1)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t


def rates(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    pos, neg = (y == 1), (y == 0)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "flag_rate": float(pred.mean()) if len(y) else float("nan"),
        "tpr": float((pred & pos).sum() / max(pos.sum(), 1)),
        "fpr": float((pred & neg).sum() / max(neg.sum(), 1)),
        "precision": float((pred & pos).sum() / max(pred.sum(), 1)),
    }


def f1_of(y, pred):
    y = np.asarray(y)
    tp = (pred & (y == 1)).sum()
    return float(2 * tp / max(pred.sum() + (y == 1).sum(), 1))


def group_table(y, pred, groups):
    out = {}
    for g in pd.unique(groups):
        m = np.asarray(groups == g)
        out[str(g)] = rates(np.asarray(y)[m], np.asarray(pred)[m])
    return out


def gap(table, key):
    vals = [v[key] for v in table.values()]
    return float(max(vals) - min(vals))


def run_seed(df, seed):
    y = df[TARGET].to_numpy()
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    sex = df["SEX"].map({1: "male", 2: "female"}).to_numpy()
    age = age_group(df["AGE"]).astype(str).to_numpy()
    res = {}
    for name, (kind, cols) in MODELS.items():
        est = make_estimator(kind, seed).fit(df.iloc[tr][cols], y[tr])
        pva = est.predict_proba(df.iloc[va][cols])[:, 1]
        pte = est.predict_proba(df.iloc[te][cols])[:, 1]
        t = best_f1_threshold(y[va], pva)
        pred = pte >= t
        sex_t = group_table(y[te], pred, sex[te])
        age_t = group_table(y[te], pred, age[te])
        # post-hoc: per-sex thresholds chosen on validation to give each sex the same flag rate as overall
        overall_rate = float((pva >= t).mean())
        thr = {g: float(np.quantile(pva[sex[va] == g], 1 - overall_rate)) for g in ("male", "female")}
        pred_eq = np.array([p >= thr[s] for p, s in zip(pte, sex[te])])
        sex_eq = group_table(y[te], pred_eq, sex[te])
        res[name] = {
            "auc": float(roc_auc_score(y[te], pte)),
            "average_precision": float(average_precision_score(y[te], pte)),
            "brier": float(brier_score_loss(y[te], pte)),
            "threshold": t,
            "f1": f1_of(y[te], pred),
            "sex": sex_t,
            "age": age_t,
            "gap_sex_flag_rate": gap(sex_t, "flag_rate"),
            "gap_sex_tpr": gap(sex_t, "tpr"),
            "gap_sex_fpr": gap(sex_t, "fpr"),
            "gap_age_flag_rate": gap(age_t, "flag_rate"),
            "gap_age_tpr": gap(age_t, "tpr"),
            "gap_age_fpr": gap(age_t, "fpr"),
            "equal_flag_rate_by_sex": {
                "f1": f1_of(y[te], pred_eq),
                "gap_flag_rate": gap(sex_eq, "flag_rate"),
                "gap_tpr": gap(sex_eq, "tpr"),
                "gap_fpr": gap(sex_eq, "fpr"),
            },
        }
    return res
