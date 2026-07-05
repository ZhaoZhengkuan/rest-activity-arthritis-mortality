#!/usr/bin/env python3
"""
Python complex-survey bootstrap for the T1 RAR mediation and Cox sensitivity
audits.

This is a PSU-within-strata bootstrap:
- within each SDMVSTRA, sample the observed SDMVPSU values with replacement;
- assign each selected PSU a multiplicity;
- multiply WTMEC4YR by the bootstrap multiplicity;
- refit the weighted component models and selected Cox models.

It is a Python substitute for design-aware uncertainty when R survey/CMAverse is
not available locally. Percentile intervals are written as audit-grade
supplemental output; final journal inference should still be cross-checked in R
when possible.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_module(filename: str, code_dir: Path | None) -> Path:
    """Locate a sibling analysis script.

    The mediation and Cox modules live next to this script (the ``code/``
    directory). They are loaded by file path because their names start with a
    digit and cannot be imported with a normal ``import`` statement. We search,
    in order: an explicit ``--code-dir``, this script's own directory, and the
    current working directory, so the bootstrap runs regardless of where
    ``--out-dir`` points.
    """
    candidates = []
    if code_dir is not None:
        candidates.append(code_dir / filename)
    candidates.append(Path(__file__).resolve().parent / filename)
    candidates.append(Path.cwd() / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate {filename}. Looked in: {searched}. "
        f"Pass --code-dir pointing at the folder that contains the analysis scripts."
    )


def bootstrap_weights(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    key = df[["SDMVSTRA", "SDMVPSU"]].drop_duplicates().copy()
    counts = []
    for strata, sub in key.groupby("SDMVSTRA", sort=False):
        psus = sub["SDMVPSU"].to_numpy()
        sampled = rng.choice(psus, size=len(psus), replace=True)
        uniq, cnt = np.unique(sampled, return_counts=True)
        counts.append(pd.DataFrame({"SDMVSTRA": strata, "SDMVPSU": uniq, "boot_mult": cnt}))
    mult = pd.concat(counts, ignore_index=True)
    out = df[["SDMVSTRA", "SDMVPSU"]].merge(mult, on=["SDMVSTRA", "SDMVPSU"], how="left")["boot_mult"].fillna(0)
    return df["WTMEC4YR"].to_numpy(dtype=float) * out.to_numpy(dtype=float)


def summarize_draws(draws: pd.DataFrame, value_col: str = "estimate") -> pd.DataFrame:
    rows = []
    for keys, sub in draws.groupby([c for c in draws.columns if c not in {"replicate", value_col}], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_cols = [c for c in draws.columns if c not in {"replicate", value_col}]
        vals = sub[value_col].dropna().to_numpy()
        row = dict(zip(key_cols, keys))
        row.update(
            {
                "n_success": int(len(vals)),
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
                "ci_lower_pct": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
                "ci_upper_pct": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="T1_RAR_arthritis_mortality_supplemental", type=Path)
    parser.add_argument(
        "--code-dir",
        default=None,
        type=Path,
        help="Folder containing the analysis scripts (03_*, 04_*). Defaults to this script's own directory.",
    )
    parser.add_argument("--n-boot", default=300, type=int)
    parser.add_argument("--seed", default=20260619, type=int)
    parser.add_argument("--tau", default=60.0, type=float)
    parser.add_argument("--run-cox-bootstrap", action="store_true", help="Also run experimental lifelines Cox bootstrap. Slow and numerically fragile.")
    args = parser.parse_args()

    out_dir = args.out_dir
    results_dir = out_dir / "results"
    docs_dir = out_dir / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    med = load_module(resolve_module("03_weighted_gformula_mediation.py", args.code_dir), "mediation_audit")
    cox = load_module(resolve_module("04_weighted_cox_sensitivity.py", args.code_dir), "cox_sensitivity")

    paths = med.Paths(out_dir)
    base_df = med.prepare_analysis_table(paths, args.tau)
    point_mediator_model, point_sigma, point_outcome_model, x_m_cols, x_y_cols = med.fit_weighted_models(base_df)
    point_effects = med.gformula_effects(base_df, point_mediator_model, point_outcome_model, x_m_cols, x_y_cols)

    if args.run_cox_bootstrap:
        cox_df = cox.load_table(out_dir)
        cox_df = cox_df.dropna(subset=[
            "iv_z_wt", "arthritis", "RIDAGEYR_z_wt", "female", "BMXBMI_z_wt",
            "hypertension_self_report", "diabetes_self_report", "cvd_history",
            "phq9_score_z_wt", "self_report_sleep_hours_z_wt", "pfq_score_v2_z_wt",
            "WTMEC4YR", "SDMVPSU", "SDMVSTRA", "permth_exm", "mortstat"
        ]).copy()
    else:
        cox_df = pd.DataFrame()

    rng = np.random.default_rng(args.seed)
    med_draws = []
    cox_draws = []
    fail_rows = []

    for b in range(args.n_boot):
        try:
            boot_df = base_df.copy()
            boot_df["WTMEC4YR"] = bootstrap_weights(boot_df, rng)
            boot_df = boot_df[boot_df["WTMEC4YR"] > 0].copy()
            m_model, sigma, y_model, xm, xy = med.fit_weighted_models(boot_df)
            eff = med.gformula_effects(boot_df, m_model, y_model, xm, xy)
            eff["replicate"] = b
            med_draws.append(eff)
        except Exception as exc:
            fail_rows.append({"replicate": b, "analysis": "gformula", "error": repr(exc)})

        if args.run_cox_bootstrap:
            try:
                boot_cox_df = cox_df.copy()
                boot_cox_df["WTMEC4YR"] = bootstrap_weights(boot_cox_df, rng)
                boot_cox_df = boot_cox_df[boot_cox_df["WTMEC4YR"] > 0].copy()
                for lag in [0, 24, 36, 60]:
                    frame = cox.model_frame(boot_cox_df, "iv_z_wt", "mortstat", lag, True)
                    res = cox.fit_one(frame, "iv_z_wt")
                    cox_draws.append({
                        "replicate": b,
                        "analysis": "lag_gradient_all_cause",
                        "outcome": "all_cause",
                        "lag_months": lag,
                        "adjust_pfq": True,
                        "estimate": res["hr"],
                    })
            except Exception as exc:
                fail_rows.append({"replicate": b, "analysis": "cox", "error": repr(exc)})

        if (b + 1) % 25 == 0:
            print(f"bootstrap {b + 1}/{args.n_boot}", flush=True)

    med_draws_df = pd.concat(med_draws, ignore_index=True) if med_draws else pd.DataFrame()
    cox_draws_df = pd.DataFrame(cox_draws)
    fail_df = pd.DataFrame(fail_rows)

    med_draws_df.to_csv(results_dir / "python_design_bootstrap_gformula_draws.csv", index=False)
    cox_draws_df.to_csv(results_dir / "python_design_bootstrap_cox_draws.csv", index=False)
    fail_df.to_csv(results_dir / "python_design_bootstrap_failures.csv", index=False)

    if not med_draws_df.empty:
        med_summary = summarize_draws(med_draws_df[["replicate", "final_standardization", "contrast", "estimate"]])
        point = point_effects.rename(columns={"estimate": "point_estimate"})
        med_summary = med_summary.merge(point, on=["final_standardization", "contrast"], how="left")
    else:
        med_summary = pd.DataFrame()
    med_summary.to_csv(results_dir / "python_design_bootstrap_gformula_ci.csv", index=False)

    if not cox_draws_df.empty:
        cox_summary = summarize_draws(cox_draws_df)
        point_cox = pd.read_csv(results_dir / "weighted_cox_sensitivity_results.csv")
        point_cox = point_cox[
            (point_cox["analysis"] == "lag_gradient_all_cause")
            & (point_cox["outcome"] == "all_cause")
            & (point_cox["adjust_pfq"] == True)
        ][["analysis", "outcome", "lag_months", "adjust_pfq", "hr"]].rename(columns={"hr": "point_estimate"})
        cox_summary = cox_summary.merge(point_cox, on=["analysis", "outcome", "lag_months", "adjust_pfq"], how="left")
    else:
        cox_summary = pd.DataFrame()
    cox_summary.to_csv(results_dir / "python_design_bootstrap_cox_ci.csv", index=False)

    manifest = {
        "script": "06_python_design_bootstrap.py",
        "n_boot_requested": args.n_boot,
        "gformula_successful_replicates": int(med_draws_df["replicate"].nunique()) if not med_draws_df.empty else 0,
        "cox_bootstrap_requested": bool(args.run_cox_bootstrap),
        "cox_successful_replicates": int(cox_draws_df["replicate"].nunique()) if not cox_draws_df.empty else 0,
        "failure_rows": int(len(fail_df)),
        "bootstrap_design": "PSU-within-SDMVSTRA resampling; WTMEC4YR multiplied by PSU selection multiplicity",
        "outputs": [
            "python_design_bootstrap_gformula_ci.csv",
            "python_design_bootstrap_cox_ci.csv",
            "python_design_bootstrap_failures.csv",
        ],
    }
    (results_dir / "python_design_bootstrap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (docs_dir / "python_design_bootstrap_notes.md").write_text(
        "# Python Design Bootstrap Notes\n\n"
        "This audit uses PSU-within-strata resampling and percentile intervals. "
        "It provides a Python-only design-aware uncertainty layer when R survey/CMAverse is unavailable locally.\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2), flush=True)
    if not med_summary.empty:
        print("\nG-formula CI:", med_summary.to_string(index=False), flush=True)
    if not cox_summary.empty:
        print("\nCox CI:", cox_summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
