#!/usr/bin/env python3
"""
Arm 1: cross-sectional survey-weighted association of arthritis with
accelerometer-derived rest-activity rhythm features.

Exposure:
    arthritis, MCQ160A=1 versus no arthritis.

Outcomes:
    iv_z, ra_z, is_z, m10_z, l5_z, m10_start_clock_z, l5_start_clock_z.

Models:
    M0: arthritis only
    M1: + age, sex, race/ethnicity, education, PIR, cycle
    M2: + BMI, smoking, hypertension, diabetes, CVD history, PHQ-9, sleep hours
    M3: + PFQ v2 functional limitation score

Weights:
    WTMEC4YR = WTMEC2YR / 2.

Inference:
    Weighted least squares with PSU-cluster robust standard errors. This is a
    Python executable substitute for survey-weighted linear regression when R
    survey is unavailable locally.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


OUTCOMES = {
    "iv_z": {"source": "iv", "label": "Intradaily variability (IV)", "expected_arthritis_direction": "higher means more fragmented"},
    "ra_z": {"source": "ra", "label": "Relative amplitude (RA)", "expected_arthritis_direction": "lower means blunted rhythm"},
    "is_z": {"source": "is", "label": "Interdaily stability (IS)", "expected_arthritis_direction": "lower means less regular rhythm"},
    "m10_z": {"source": "m10", "label": "M10 activity level", "expected_arthritis_direction": "lower means lower peak daytime activity"},
    "l5_z": {"source": "l5", "label": "L5 activity level", "expected_arthritis_direction": "higher may indicate less consolidated rest"},
    "m10_start_clock_z": {"source": "m10_start_clock", "label": "M10 start time", "expected_arthritis_direction": "later/earlier timing shift"},
    "l5_start_clock_z": {"source": "l5_start_clock", "label": "L5 start time", "expected_arthritis_direction": "later/earlier timing shift"},
}


MODEL_COVARS = {
    "M0_unadjusted": [],
    "M1_demographic": ["RIDAGEYR_z_wt", "female", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "cycle"],
    "M2_clinical": [
        "RIDAGEYR_z_wt",
        "female",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
        "cycle",
        "BMXBMI_z_wt",
        "smoking_status",
        "hypertension_self_report",
        "diabetes_self_report",
        "cvd_history",
        "phq9_score_z_wt",
        "self_report_sleep_hours_z_wt",
    ],
    "M3_plus_PFQ": [
        "RIDAGEYR_z_wt",
        "female",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
        "cycle",
        "BMXBMI_z_wt",
        "smoking_status",
        "hypertension_self_report",
        "diabetes_self_report",
        "cvd_history",
        "phq9_score_z_wt",
        "self_report_sleep_hours_z_wt",
        "pfq_score_v2_z_wt",
    ],
}


def weighted_mean_sd(x: pd.Series, w: pd.Series) -> tuple[float, float]:
    mask = x.notna() & w.notna() & (w > 0)
    xv = x[mask].astype(float).to_numpy()
    wv = w[mask].astype(float).to_numpy()
    mu = np.average(xv, weights=wv)
    sd = math.sqrt(np.average((xv - mu) ** 2, weights=wv))
    return float(mu), float(sd)


def clock_center(minutes: pd.Series) -> pd.Series:
    """Represent circular start time as signed hours around noon-ish median.

    For simple Arm 1 timing screening, transform minutes to hours since midnight
    and use linear z-scoring. The raw minute columns remain in the output table.
    """
    return minutes.astype(float) / 60.0


def build_arm1_table(root: Path, out_dir: Path) -> pd.DataFrame:
    base = pd.read_parquet(root / "data" / "t1_analysis_base_day_pfq_mortality.parquet")
    feat = pd.read_parquet(root / "data" / "paxmin_rar_features_2011_2014.parquet")
    df = base.merge(feat, on=["SEQN", "cycle"], how="inner", suffixes=("", "_paxmin"))
    df = df[
        (df["adult20"] == True)
        & (df["pax_has_4_valid_days_v2"] == True)
        & (df["feature_status"] == "ok")
    ].copy()
    df = df.dropna(subset=["arthritis", "WTMEC4YR", "SDMVSTRA", "SDMVPSU"]).copy()
    df["arthritis"] = df["arthritis"].astype(int)
    df["m10_start_clock"] = clock_center(df["m10_start_minute"])
    df["l5_start_clock"] = clock_center(df["l5_start_minute"])
    for out, meta in OUTCOMES.items():
        src = meta["source"]
        mu, sd = weighted_mean_sd(df[src], df["WTMEC4YR"])
        df[out] = (df[src] - mu) / sd
    out_dir.joinpath("data").mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "data" / "arm1_analysis_table.parquet", index=False)
    df.to_csv(out_dir / "data" / "arm1_analysis_table.csv", index=False)
    return df


def design_matrix(df: pd.DataFrame, covars: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["intercept"] = 1.0
    x["arthritis"] = df["arthritis"].astype(float)
    categorical = {"RIDRETH3", "DMDEDUC2", "smoking_status", "cycle"}
    for col in covars:
        if col in categorical:
            dummies = pd.get_dummies(df[col].astype("category"), prefix=col, drop_first=True, dtype=float)
            x = pd.concat([x, dummies], axis=1)
        else:
            x[col] = df[col].astype(float)
    return x


def fit_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    desc_rows = []
    for outcome, meta in OUTCOMES.items():
        for arm_val, arm_name in [(0, "No arthritis"), (1, "Arthritis")]:
            sub = df[df["arthritis"] == arm_val]
            mu, sd = weighted_mean_sd(sub[outcome], sub["WTMEC4YR"])
            raw_mu, raw_sd = weighted_mean_sd(sub[meta["source"]], sub["WTMEC4YR"])
            desc_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": meta["label"],
                    "group": arm_name,
                    "n": int(len(sub)),
                    "weighted_mean_z": mu,
                    "weighted_sd_z": sd,
                    "weighted_mean_raw": raw_mu,
                    "weighted_sd_raw": raw_sd,
                }
            )

        for model_name, covars in MODEL_COVARS.items():
            needed = [outcome, "arthritis", "WTMEC4YR", "SDMVPSU"] + covars
            use = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
            x = design_matrix(use, covars)
            y = use[outcome].astype(float)
            w = use["WTMEC4YR"].astype(float)
            fit = sm.WLS(y, x, weights=w).fit(cov_type="cluster", cov_kwds={"groups": use["SDMVPSU"]})
            coef = fit.params.get("arthritis", np.nan)
            se = fit.bse.get("arthritis", np.nan)
            p = fit.pvalues.get("arthritis", np.nan)
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": meta["label"],
                    "expected_direction_note": meta["expected_arthritis_direction"],
                    "model": model_name,
                    "n": int(len(use)),
                    "arthritis_n": int(use["arthritis"].sum()),
                    "beta_z": coef,
                    "se": se,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "p": p,
                }
            )
    res = pd.DataFrame(rows)
    res["q_fdr_within_model"] = np.nan
    for model_name, sub_idx in res.groupby("model").groups.items():
        pvals = res.loc[sub_idx, "p"].to_numpy()
        ok = np.isfinite(pvals)
        q = np.full(len(pvals), np.nan)
        if ok.sum() > 0:
            q[ok] = multipletests(pvals[ok], method="fdr_bh")[1]
        res.loc[sub_idx, "q_fdr_within_model"] = q
    return res, pd.DataFrame(desc_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="T1_RAR_arthritis_mortality_supplemental",
        type=Path,
        help="Folder that holds the input data/ directory (base table + RAR features).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Where to write Arm 1 outputs. Defaults to --root so the derived arm1 table "
        "lands in the same data/ folder as the other tables.",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.root

    for sub in ["data", "results", "docs"]:
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    df = build_arm1_table(args.root, args.out_dir)
    res, desc = fit_models(df)
    res.to_csv(args.out_dir / "results" / "arm1_weighted_arthritis_rar_regression.csv", index=False)
    desc.to_csv(args.out_dir / "results" / "arm1_weighted_group_descriptives.csv", index=False)

    main = res[res["model"] == "M3_plus_PFQ"].copy()
    main.to_csv(args.out_dir / "results" / "arm1_main_model_plus_pfq.csv", index=False)

    manifest = {
        "script": "arm1_weighted_arthritis_rar.py",
        "analysis_n": int(len(df)),
        "arthritis_n": int(df["arthritis"].sum()),
        "no_arthritis_n": int((df["arthritis"] == 0).sum()),
        "primary_model": "M2_clinical for main cross-sectional phenotype difference; M3_plus_PFQ as attenuation/functional-limitation sensitivity",
        "outcomes": list(OUTCOMES.keys()),
        "weights": "WTMEC4YR",
        "inference": "WLS with PSU-cluster robust SE",
    }
    (args.out_dir / "results" / "arm1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("M2 clinical model:")
    print(res[res["model"] == "M2_clinical"][["outcome", "beta_z", "ci_lower", "ci_upper", "p", "q_fdr_within_model", "n"]].to_string(index=False))
    print("\nM3 plus PFQ:")
    print(main[["outcome", "beta_z", "ci_lower", "ci_upper", "p", "q_fdr_within_model", "n"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
