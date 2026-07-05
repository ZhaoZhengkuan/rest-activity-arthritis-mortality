#!/usr/bin/env python3
"""
Python fallback/audit implementation for the T1 mediation analysis.

Purpose
-------
This script addresses the practical gap created by the unavailable R/CMAverse
environment. It does not claim to replace CMAverse for the final manuscript.
Instead, it provides an auditable weighted g-formula check that:

1. uses one locked mediator: iv_z_wt
2. fits all component models with MEC survey weights
3. reports both unweighted and MEC-weighted final marginal standardization
4. compares weighted vs unweighted final standardization explicitly
5. provides a weighted Cox sanity check for the mediator-outcome model

Estimand
--------
The main estimand is a fixed-horizon interventional g-formula decomposition on
the 60-month mortality risk-difference scale. Participants censored before 60
months without death are excluded from the fixed-horizon risk set.

This is a conservative Python fallback. The preferred final analysis remains
CMAverse interventional effects for survival outcomes in R, ideally with the
weighted marginal standardization audit implemented here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter


LOCKED_MEDIATOR = "iv_z_wt"


@dataclass
class Paths:
    out_dir: Path

    @property
    def data(self) -> Path:
        return self.out_dir / "data"

    @property
    def results(self) -> Path:
        return self.out_dir / "results"

    @property
    def docs(self) -> Path:
        return self.out_dir / "docs"


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[mask], weights=w[mask]))


def prepare_analysis_table(paths: Paths, tau: float) -> pd.DataFrame:
    base = pd.read_parquet(paths.data / "t1_analysis_base_day_pfq_mortality.parquet")
    feat = pd.read_parquet(paths.data / "paxmin_rar_features_2011_2014.parquet")
    df = base.merge(feat, on=["SEQN", "cycle"], how="inner", suffixes=("", "_paxmin"))
    df = df[
        (df["adult20"] == True)
        & (df["eligible_mortality"] == True)
        & (df["pax_has_4_valid_days_v2"] == True)
        & (df["feature_status"] == "ok")
    ].copy()
    df["event_by_tau"] = np.where((df["mortstat"] == 1) & (df["permth_exm"] <= tau), 1.0, 0.0)
    df["known_tau_status"] = (df["permth_exm"] >= tau) | ((df["mortstat"] == 1) & (df["permth_exm"] <= tau))
    df = df[df["known_tau_status"]].copy()

    # Conservative complete-case set for this audit.
    covars = [
        "arthritis",
        LOCKED_MEDIATOR,
        "pfq_score_v2_z_wt",
        "RIDAGEYR_z_wt",
        "female",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
        "BMXBMI_z_wt",
        "smoking_status",
        "hypertension_self_report",
        "diabetes_self_report",
        "cvd_history",
        "phq9_score_z_wt",
        "self_report_sleep_hours_z_wt",
        "WTMEC4YR",
        "SDMVPSU",
        "SDMVSTRA",
        "event_by_tau",
        "permth_exm",
        "mortstat",
    ]
    covars = [c for c in covars if c in df.columns]
    df = df.dropna(subset=covars).copy()
    df["arthritis"] = df["arthritis"].astype(int)
    df["female"] = df["female"].astype(int)
    return df


def design_matrix(df: pd.DataFrame, include_mediator: bool, include_interaction: bool, include_pfq: bool = True) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["intercept"] = 1.0
    x["arthritis"] = df["arthritis"].astype(float)
    if include_mediator:
        x[LOCKED_MEDIATOR] = df[LOCKED_MEDIATOR].astype(float)
        if include_interaction:
            x[f"arthritis_x_{LOCKED_MEDIATOR}"] = x["arthritis"] * x[LOCKED_MEDIATOR]
    if include_pfq:
        x["pfq_score_v2_z_wt"] = df["pfq_score_v2_z_wt"].astype(float)
    for col in ["RIDAGEYR_z_wt", "female", "INDFMPIR", "BMXBMI_z_wt", "hypertension_self_report", "diabetes_self_report", "cvd_history", "phq9_score_z_wt", "self_report_sleep_hours_z_wt"]:
        if col in df.columns:
            x[col] = df[col].astype(float)
    for col in ["RIDRETH3", "DMDEDUC2", "smoking_status", "cycle"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype("category"), prefix=col, drop_first=True, dtype=float)
            x = pd.concat([x, dummies], axis=1)
    return x


def fit_weighted_models(df: pd.DataFrame):
    w = df["WTMEC4YR"].astype(float)
    x_m = design_matrix(df, include_mediator=False, include_interaction=False, include_pfq=True)
    mediator_model = sm.WLS(df[LOCKED_MEDIATOR].astype(float), x_m, weights=w).fit()
    mediator_sigma = math.sqrt(np.average(mediator_model.resid**2, weights=w.loc[x_m.index]))

    x_y = design_matrix(df, include_mediator=True, include_interaction=True, include_pfq=True)
    outcome_model = sm.GLM(
        df["event_by_tau"].astype(float),
        x_y,
        family=sm.families.Binomial(),
        freq_weights=w,
    ).fit(maxiter=200)
    return mediator_model, mediator_sigma, outcome_model, x_m.columns.tolist(), x_y.columns.tolist()


def predict_mediator(df: pd.DataFrame, mediator_model, a: int, x_m_cols: list[str]) -> np.ndarray:
    tmp = df.copy()
    tmp["arthritis"] = a
    x = design_matrix(tmp, include_mediator=False, include_interaction=False, include_pfq=True)
    x = x.reindex(columns=x_m_cols, fill_value=0.0)
    return np.asarray(mediator_model.predict(x), dtype=float)


def predict_risk(df: pd.DataFrame, outcome_model, a: int, m_values: np.ndarray, x_y_cols: list[str]) -> np.ndarray:
    tmp = df.copy()
    tmp["arthritis"] = a
    tmp[LOCKED_MEDIATOR] = m_values
    x = design_matrix(tmp, include_mediator=True, include_interaction=True, include_pfq=True)
    x = x.reindex(columns=x_y_cols, fill_value=0.0)
    return np.asarray(outcome_model.predict(x), dtype=float)


def gformula_effects(df: pd.DataFrame, mediator_model, outcome_model, x_m_cols: list[str], x_y_cols: list[str]) -> pd.DataFrame:
    rows = []
    w = df["WTMEC4YR"].to_numpy(dtype=float)
    for final_weighting, weights in [("analysis_sample_unweighted", np.ones(len(df))), ("mec_weighted", w)]:
        m0 = predict_mediator(df, mediator_model, 0, x_m_cols)
        m1 = predict_mediator(df, mediator_model, 1, x_m_cols)
        y00 = predict_risk(df, outcome_model, 0, m0, x_y_cols)
        y10 = predict_risk(df, outcome_model, 1, m0, x_y_cols)
        y11 = predict_risk(df, outcome_model, 1, m1, x_y_cols)
        r00 = weighted_mean(y00, weights)
        r10 = weighted_mean(y10, weights)
        r11 = weighted_mean(y11, weights)
        total = r11 - r00
        direct = r10 - r00
        indirect = r11 - r10
        rows.extend(
            [
                {"final_standardization": final_weighting, "contrast": "risk_A0_M0", "estimate": r00},
                {"final_standardization": final_weighting, "contrast": "risk_A1_M0", "estimate": r10},
                {"final_standardization": final_weighting, "contrast": "risk_A1_M1", "estimate": r11},
                {"final_standardization": final_weighting, "contrast": "total_effect_rd", "estimate": total},
                {"final_standardization": final_weighting, "contrast": "interventional_direct_effect_rd", "estimate": direct},
                {"final_standardization": final_weighting, "contrast": "interventional_indirect_effect_rd", "estimate": indirect},
                {
                    "final_standardization": final_weighting,
                    "contrast": "proportion_mediated_rd",
                    "estimate": indirect / total if abs(total) > 1e-12 else np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def fit_weighted_cox(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    # Cox sanity check, not the mediation estimator. lifelines treats weights as
    # case weights and uses robust variance clustered by PSU.
    cdf = df.copy()
    cols = [
        "permth_exm",
        "mortstat",
        "WTMEC4YR",
        "SDMVPSU",
        "arthritis",
        LOCKED_MEDIATOR,
        "pfq_score_v2_z_wt",
        "RIDAGEYR_z_wt",
        "female",
        "BMXBMI_z_wt",
        "hypertension_self_report",
        "diabetes_self_report",
        "cvd_history",
        "phq9_score_z_wt",
        "self_report_sleep_hours_z_wt",
    ]
    cdf = cdf[cols].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(
        cdf,
        duration_col="permth_exm",
        event_col="mortstat",
        weights_col="WTMEC4YR",
        cluster_col="SDMVPSU",
        robust=True,
    )
    out = cph.summary.reset_index().rename(columns={"covariate": "term"})
    out.to_csv(paths.results / "weighted_cox_sanity_check.csv", index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="T1_RAR_arthritis_mortality_supplemental", type=Path)
    parser.add_argument("--tau", default=60.0, type=float)
    args = parser.parse_args()

    paths = Paths(args.out_dir)
    for p in [paths.results, paths.docs]:
        p.mkdir(parents=True, exist_ok=True)

    df = prepare_analysis_table(paths, args.tau)
    mediator_model, mediator_sigma, outcome_model, x_m_cols, x_y_cols = fit_weighted_models(df)
    effects = gformula_effects(df, mediator_model, outcome_model, x_m_cols, x_y_cols)
    effects.to_csv(paths.results / "weighted_gformula_mediation_60m.csv", index=False)

    model_rows = []
    for term, val in mediator_model.params.items():
        model_rows.append({"model": "weighted_linear_mediator", "term": term, "estimate": val, "p_value": mediator_model.pvalues.get(term, np.nan)})
    for term, val in outcome_model.params.items():
        model_rows.append({"model": "weighted_logistic_60m_outcome", "term": term, "estimate": val, "p_value": outcome_model.pvalues.get(term, np.nan)})
    model_rows.append({"model": "weighted_linear_mediator", "term": "weighted_residual_sigma", "estimate": mediator_sigma, "p_value": np.nan})
    pd.DataFrame(model_rows).to_csv(paths.results / "weighted_gformula_component_models.csv", index=False)

    cox = fit_weighted_cox(df, paths)

    wide = effects.pivot(index="contrast", columns="final_standardization", values="estimate").reset_index()
    if {"analysis_sample_unweighted", "mec_weighted"}.issubset(wide.columns):
        wide["mec_minus_unweighted"] = wide["mec_weighted"] - wide["analysis_sample_unweighted"]
    wide.to_csv(paths.results / "weighted_vs_unweighted_standardization_audit.csv", index=False)

    manifest = {
        "script": "03_weighted_gformula_mediation.py",
        "locked_primary_mediator": LOCKED_MEDIATOR,
        "tau_months": args.tau,
        "analysis_n": int(len(df)),
        "events_by_tau": int(df["event_by_tau"].sum()),
        "all_cause_events_all_followup": int(df["mortstat"].sum()),
        "component_models_weighted": True,
        "final_standardization_outputs": ["analysis_sample_unweighted", "mec_weighted"],
        "note": "Python fixed-horizon risk-difference g-formula fallback. Preferred final manuscript analysis remains R/CMAverse interventional survival mediation.",
    }
    (paths.results / "weighted_gformula_mediation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = [
        "# Weighted G-Formula Mediation Audit",
        "",
        f"Analysis N: {len(df)}; deaths by {args.tau:.0f} months: {int(df['event_by_tau'].sum())}.",
        "",
        "Locked mediator: `iv_z_wt`.",
        "",
        "All component models used `WTMEC4YR`; final standardization is reported both unweighted and MEC-weighted.",
        "",
        "Key output files:",
        "",
        "- `weighted_gformula_mediation_60m.csv`",
        "- `weighted_vs_unweighted_standardization_audit.csv`",
        "- `weighted_gformula_component_models.csv`",
        "- `weighted_cox_sanity_check.csv`",
        "",
        "Interpretation boundary: this is a Python audit/fallback on a fixed-horizon risk-difference scale, not a replacement for the planned CMAverse interventional survival mediation.",
    ]
    (paths.docs / "weighted_gformula_mediation_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(effects.to_string(index=False), flush=True)
    print("\nCox sanity terms:", cox.loc[cox["term"].isin(["arthritis", LOCKED_MEDIATOR]), ["term", "coef", "exp(coef)", "p"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
