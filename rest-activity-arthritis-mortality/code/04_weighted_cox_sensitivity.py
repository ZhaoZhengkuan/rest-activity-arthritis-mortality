#!/usr/bin/env python3
"""
Weighted Cox sensitivity analyses for the T1 RAR/arthritis/mortality package.

Analyses:
- IV_z_wt -> all-cause mortality with and without PFQ adjustment
- lag-gradient exclusions: deaths within 0, 24, 36, 60 months
- cause-specific mortality: CVD = UCOD 001 + 005, cancer, respiratory
- deterministic shuffled-IV pipeline calibration

These are case-weighted Cox models using lifelines, robust variance clustered
by SDMVPSU. R survey::svycoxph remains preferred for the final manuscript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


MEDIATOR = "iv_z_wt"


def load_table(out_dir: Path) -> pd.DataFrame:
    base = pd.read_parquet(out_dir / "data" / "t1_analysis_base_day_pfq_mortality.parquet")
    feat = pd.read_parquet(out_dir / "data" / "paxmin_rar_features_2011_2014.parquet")
    df = base.merge(feat, on=["SEQN", "cycle"], how="inner", suffixes=("", "_paxmin"))
    df = df[
        (df["adult20"] == True)
        & (df["eligible_mortality"] == True)
        & (df["pax_has_4_valid_days_v2"] == True)
        & (df["feature_status"] == "ok")
    ].copy()
    return df


def model_frame(df: pd.DataFrame, exposure: str, outcome: str, lag_months: float, adjust_pfq: bool) -> pd.DataFrame:
    sub = df.copy()
    if lag_months > 0:
        early_death = (sub["mortstat"] == 1) & (sub["permth_exm"] <= lag_months)
        sub = sub[~early_death].copy()
        sub["time"] = sub["permth_exm"] - lag_months
        sub = sub[sub["time"] > 0].copy()
    else:
        sub["time"] = sub["permth_exm"]
    sub["event"] = sub[outcome].astype(float)
    covars = [
        exposure,
        "arthritis",
        "RIDAGEYR_z_wt",
        "female",
        "BMXBMI_z_wt",
        "hypertension_self_report",
        "diabetes_self_report",
        "cvd_history",
        "phq9_score_z_wt",
        "self_report_sleep_hours_z_wt",
        "WTMEC4YR",
        "SDMVPSU",
        "time",
        "event",
    ]
    if adjust_pfq:
        covars.append("pfq_score_v2_z_wt")
    keep = sub[covars].dropna().copy()
    return keep


def fit_one(frame: pd.DataFrame, exposure: str) -> dict:
    cph = CoxPHFitter()
    cph.fit(
        frame,
        duration_col="time",
        event_col="event",
        weights_col="WTMEC4YR",
        cluster_col="SDMVPSU",
        robust=True,
    )
    s = cph.summary.loc[exposure]
    return {
        "n": int(len(frame)),
        "events": int(frame["event"].sum()),
        "term": exposure,
        "coef": float(s["coef"]),
        "hr": float(s["exp(coef)"]),
        "ci_lower": float(s["exp(coef) lower 95%"]),
        "ci_upper": float(s["exp(coef) upper 95%"]),
        "p": float(s["p"]),
    }


def e_value(hr: float, lo: float, hi: float) -> tuple[float, float]:
    rr = hr if hr >= 1 else 1 / hr
    ev = rr + np.sqrt(rr * (rr - 1)) if rr >= 1 else np.nan
    bound = lo if hr > 1 else 1 / hi
    ev_ci = bound + np.sqrt(bound * (bound - 1)) if bound > 1 else 1.0
    return float(ev), float(ev_ci)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="T1_RAR_arthritis_mortality_supplemental", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir
    results_dir = out_dir / "results"
    docs_dir = out_dir / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(out_dir)
    rng = np.random.default_rng(20260619)
    df["iv_z_wt_shuffled"] = rng.permutation(df[MEDIATOR].to_numpy())

    rows = []
    outcomes = {
        "all_cause": "mortstat",
        "cvd_001_005": "cvd_death",
        "cancer": "cancer_death",
        "respiratory": "respiratory_death",
    }
    for adjust_pfq in [False, True]:
        for lag in [0, 24, 36, 60]:
            frame = model_frame(df, MEDIATOR, "mortstat", lag, adjust_pfq)
            res = fit_one(frame, MEDIATOR)
            ev, ev_ci = e_value(res["hr"], res["ci_lower"], res["ci_upper"])
            rows.append({"analysis": "lag_gradient_all_cause", "outcome": "all_cause", "lag_months": lag, "adjust_pfq": adjust_pfq, **res, "e_value": ev, "e_value_ci_bound": ev_ci})
    for name, outcome in outcomes.items():
        frame = model_frame(df, MEDIATOR, outcome, 0, True)
        if frame["event"].sum() >= 30:
            res = fit_one(frame, MEDIATOR)
            ev, ev_ci = e_value(res["hr"], res["ci_lower"], res["ci_upper"])
            rows.append({"analysis": "cause_specific", "outcome": name, "lag_months": 0, "adjust_pfq": True, **res, "e_value": ev, "e_value_ci_bound": ev_ci})
        else:
            rows.append({"analysis": "cause_specific", "outcome": name, "lag_months": 0, "adjust_pfq": True, "n": int(len(frame)), "events": int(frame["event"].sum()), "term": MEDIATOR, "coef": np.nan, "hr": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p": np.nan, "e_value": np.nan, "e_value_ci_bound": np.nan})

    frame = model_frame(df, "iv_z_wt_shuffled", "mortstat", 0, True)
    res = fit_one(frame, "iv_z_wt_shuffled")
    rows.append({"analysis": "permuted_exposure_pipeline_calibration", "outcome": "all_cause", "lag_months": 0, "adjust_pfq": True, **res, "e_value": np.nan, "e_value_ci_bound": np.nan})

    result = pd.DataFrame(rows)
    result.to_csv(results_dir / "weighted_cox_sensitivity_results.csv", index=False)

    manifest = {
        "script": "04_weighted_cox_sensitivity.py",
        "analysis_n_before_complete_case": int(len(df)),
        "primary_exposure": MEDIATOR,
        "cvd_definition": "UCOD_LEADING 001 heart disease + 005 cerebrovascular disease",
        "permuted_exposure_note": "Pipeline calibration/fake-signal check, not a Lipsitch-style residual-confounding negative control.",
        "output": str(results_dir / "weighted_cox_sensitivity_results.csv"),
    }
    (results_dir / "weighted_cox_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = [
        "# Weighted Cox Sensitivity Results",
        "",
        "This file records Python case-weighted Cox sensitivity analyses. Final manuscript inference should be confirmed with `survey::svycoxph`.",
        "",
        "Key positioning:",
        "",
        "- `iv_z_wt_shuffled` is a pipeline calibration/fake-signal check, not a formal residual-confounding negative-control exposure.",
        "- CVD death is harmonized as UCOD 001 + 005.",
        "- Lag-gradient exclusions use 0, 24, 36, and 60 months.",
        "",
    ]
    (docs_dir / "weighted_cox_sensitivity_notes.md").write_text("\n".join(report), encoding="utf-8")
    print(result.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
