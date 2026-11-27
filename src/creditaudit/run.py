import argparse
import json
from pathlib import Path

import numpy as np

from .audit import MODELS, run_seed
from .data import ensure_data, load

SEEDS = [0, 1, 2, 3, 4]
SCALARS = ["auc", "average_precision", "brier", "f1", "gap_sex_flag_rate", "gap_sex_tpr", "gap_sex_fpr",
           "gap_age_flag_rate", "gap_age_tpr", "gap_age_fpr"]


def summarize(per_seed):
    out = {}
    for m in MODELS:
        d = {}
        for k in SCALARS:
            v = np.array([s[m][k] for s in per_seed])
            d[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)}
        eq = {}
        for k in ("f1", "gap_flag_rate", "gap_tpr", "gap_fpr"):
            v = np.array([s[m]["equal_flag_rate_by_sex"][k] for s in per_seed])
            eq[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)}
        d["equal_flag_rate_by_sex"] = eq
        out[m] = d
    return out


def run(seeds=SEEDS, subsample=None, out_dir="results", data_dir="data"):
    ensure_data(data_dir)
    df = load(data_dir)
    if subsample:
        df = df.sample(n=subsample, random_state=0).reset_index(drop=True)
    per_seed = [run_seed(df, s) for s in seeds]
    first = per_seed[0]
    result = {
        "dataset": "UCI Default of Credit Card Clients (Taiwan, 2005)",
        "n_rows_used": int(len(df)),
        "seeds": list(seeds),
        "protocol": "60/20/20 stratified train/validation/test per seed; threshold maximises F1 on validation; "
                    "all group metrics are on the test split",
        "subsample_smoke_run": bool(subsample),
        "summary": summarize(per_seed),
        "group_detail_seed0": {m: {"sex": first[m]["sex"], "age": first[m]["age"]} for m in MODELS},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    plot(result, out / "group_gaps.png")
    return result


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(result["summary"])
    keys = [("gap_sex_tpr", "TPR gap (sex)"), ("gap_sex_fpr", "FPR gap (sex)"),
            ("gap_age_tpr", "TPR gap (age)"), ("gap_age_fpr", "FPR gap (age)")]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.2
    for i, (k, label) in enumerate(keys):
        vals = [result["summary"][n][k]["mean"] for n in names]
        ax.bar(np.arange(len(names)) + i * w, vals, w, label=label)
    ax.set_xticks(np.arange(len(names)) + 1.5 * w)
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("max minus min across groups (mean of seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    r = run(SEEDS[: a.seeds], a.subsample, a.out)
    for m, s in r["summary"].items():
        print(f"{m:20s} auc={s['auc']['mean']:.4f} f1={s['f1']['mean']:.4f} sexTPRgap={s['gap_sex_tpr']['mean']:.4f}")


if __name__ == "__main__":
    main()
