import numpy as np
import pandas as pd
import pytest

from creditaudit.audit import best_f1_threshold, f1_of, gap, group_table, rates, run_seed
from creditaudit.data import SHA256, TARGET, age_group, ensure_data, load, sha256_of
from creditaudit.run import run


def test_rates_hand_computed():
    y = np.array([1, 1, 0, 0, 0])
    pred = np.array([True, False, True, False, False])
    r = rates(y, pred)
    assert r["tpr"] == 0.5 and r["fpr"] == pytest.approx(1 / 3) and r["precision"] == 0.5
    assert r["flag_rate"] == pytest.approx(0.4) and r["base_rate"] == pytest.approx(0.4)


def test_group_table_and_gap():
    y = np.array([1, 0, 1, 0])
    pred = np.array([True, False, False, False])
    g = np.array(["a", "a", "b", "b"])
    t = group_table(y, pred, g)
    assert t["a"]["tpr"] == 1.0 and t["b"]["tpr"] == 0.0
    assert gap(t, "tpr") == 1.0


def test_threshold_and_f1():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.7] * 50)
    t = best_f1_threshold(y, p)
    assert 0.1 < t <= 0.7
    assert f1_of(y, p >= t) == 1.0


def test_age_groups():
    g = age_group(pd.Series([21, 29, 30, 39, 40, 79])).astype(str).tolist()
    assert g == ["<=29", "<=29", "30-39", "30-39", ">=40", ">=40"]


def test_checksum_pin_and_real_shape():
    d = ensure_data("data")
    assert sha256_of(d / "credit.zip") == SHA256
    df = load(d)
    assert df.shape == (30000, 25) and df[TARGET].sum() == 6636


def test_run_seed_structure_on_sample():
    df = load("data").sample(n=8000, random_state=0).reset_index(drop=True)
    r = run_seed(df, 0)
    assert set(r) == {"logreg_without_sex", "logreg_with_sex", "hgb_without_sex", "hgb_with_sex"}
    for m in r.values():
        assert 0.5 < m["auc"] < 1.0
        assert set(m["sex"]) == {"male", "female"}
        assert m["equal_flag_rate_by_sex"]["gap_flag_rate"] < 0.05


def test_smoke_run_writes_outputs(tmp_path):
    run(seeds=[0], subsample=2000, out_dir=tmp_path)
    assert (tmp_path / "results.json").exists() and (tmp_path / "group_gaps.png").exists()
