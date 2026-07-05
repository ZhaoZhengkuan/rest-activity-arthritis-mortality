#!/usr/bin/env python3
"""
Build the person-level analysis base table for the NHANES 2011-2014
arthritis, accelerometer rest-activity rhythm, physical functioning, and
mortality project.

This script intentionally does not compute PAXMIN-derived RAR features. It
creates the stable person-level scaffold that the minute-level feature script
can merge onto:

- survey design variables and 4-year MEC weights
- arthritis exposure
- PFQ v2 functional limitation score with explicit handling of codes 5/7/9
- day-level accelerometer feasibility features from PAXDAY
- core covariates
- public-use 2019 linked mortality outcomes

The 2022 NHANES linked mortality follow-up is restricted-use at the time this
pipeline was written; the public 2019 files are downloaded only as an
executable fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyreadstat


CYCLES = {
    "G": {"cycle": "2011_2012", "label": "2011-2012", "mort": "NHANES_2011_2012_MORT_2019_PUBLIC.dat"},
    "H": {"cycle": "2013_2014", "label": "2013-2014", "mort": "NHANES_2013_2014_MORT_2019_PUBLIC.dat"},
}

MORTALITY_URLS = {
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2013_2014_MORT_2019_PUBLIC.dat",
}

PFQ061_ITEMS = [f"PFQ061{x}" for x in list("ABCDEFGHIJKLMNOPQRST")]
PFQ_BROAD_ITEMS = ["PFQ049", "PFQ051", "PFQ057", "PFQ059"]


@dataclass
class Paths:
    nhanes_root: Path
    out_dir: Path

    @property
    def raw(self) -> Path:
        return self.nhanes_root / "raw"

    @property
    def external_mortality(self) -> Path:
        return self.out_dir / "external" / "mortality_2019_public"

    @property
    def data(self) -> Path:
        return self.out_dir / "data"

    @property
    def results(self) -> Path:
        return self.out_dir / "results"

    @property
    def docs(self) -> Path:
        return self.out_dir / "docs"


def read_xpt(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df, _ = pyreadstat.read_xport(str(path), usecols=list(columns) if columns else None)
    return df


def read_module(paths: Paths, suffix: str, module: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cycle = CYCLES[suffix]["cycle"]
    path = paths.raw / cycle / f"{module}_{suffix}.XPT"
    return read_xpt(path, columns)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def valid_yes_no(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[series == 1] = 1.0
    out[series == 2] = 0.0
    return out


def phq9_score(dpq: pd.DataFrame) -> pd.DataFrame:
    items = [f"DPQ{x:03d}" for x in range(10, 100, 10)]
    keep = ["SEQN"] + [c for c in items if c in dpq.columns]
    out = dpq[keep].copy()
    for col in items:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[~out[col].isin([0, 1, 2, 3]), col] = np.nan
    item_cols = [c for c in items if c in out.columns]
    out["phq9_valid_items"] = out[item_cols].notna().sum(axis=1)
    raw = out[item_cols].sum(axis=1, min_count=1)
    out["phq9_score"] = np.where(out["phq9_valid_items"] >= 7, raw * 9 / out["phq9_valid_items"], np.nan)
    return out[["SEQN", "phq9_score", "phq9_valid_items"]]


def pfq_score_v2(pfq: pd.DataFrame) -> pd.DataFrame:
    keep = ["SEQN"] + [c for c in PFQ061_ITEMS + PFQ_BROAD_ITEMS if c in pfq.columns]
    out = pfq[keep].copy()
    for col in keep:
        if col != "SEQN":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    score_parts = []
    valid_parts = []
    do_not_do_flags = []

    for col in PFQ061_ITEMS:
        if col not in out.columns:
            continue
        raw = out[col]
        positive = raw.isin([2, 3, 4]).astype(float)
        valid = raw.isin([1, 2, 3, 4]).astype(float)
        if "PFQ059" in out.columns:
            structural_zero = raw.isna() & (out["PFQ059"] == 2)
            positive[structural_zero] = 0.0
            valid[structural_zero] = 1.0
        positive[valid == 0] = np.nan
        score_parts.append(positive)
        valid_parts.append(valid.replace({0: np.nan}))
        do_not_do_flags.append((raw == 5).astype(int))

    for col in PFQ_BROAD_ITEMS:
        if col not in out.columns:
            continue
        raw = out[col]
        positive = (raw == 1).astype(float)
        valid = raw.isin([1, 2]).astype(float)
        positive[valid == 0] = np.nan
        score_parts.append(positive)
        valid_parts.append(valid.replace({0: np.nan}))

    if score_parts:
        score_mat = pd.concat(score_parts, axis=1)
        valid_mat = pd.concat(valid_parts, axis=1)
        out["pfq_score_v2_raw"] = score_mat.sum(axis=1, min_count=1)
        out["pfq_score_v2_valid_items"] = valid_mat.notna().sum(axis=1)
        # Primary score: prorate to the prespecified 24-item maximum if at
        # least 75% of candidate items are valid. Codes 5/7/9 are not treated
        # as high-difficulty responses.
        out["pfq_score_v2"] = np.where(
            out["pfq_score_v2_valid_items"] >= 18,
            out["pfq_score_v2_raw"] * 24.0 / out["pfq_score_v2_valid_items"],
            np.nan,
        )
    else:
        out["pfq_score_v2_raw"] = np.nan
        out["pfq_score_v2_valid_items"] = 0
        out["pfq_score_v2"] = np.nan

    if do_not_do_flags:
        out["pfq061_do_not_do_count"] = pd.concat(do_not_do_flags, axis=1).sum(axis=1)
    else:
        out["pfq061_do_not_do_count"] = np.nan

    return out[
        [
            "SEQN",
            "pfq_score_v2",
            "pfq_score_v2_raw",
            "pfq_score_v2_valid_items",
            "pfq061_do_not_do_count",
        ]
    ]


def build_paxday_features(paths: Paths, suffix: str) -> pd.DataFrame:
    pax = read_module(
        paths,
        suffix,
        "PAXDAY",
        [
            "SEQN",
            "PAXDAYD",
            "PAXDAYWD",
            "PAXTMD",
            "PAXVMD",
            "PAXMTSD",
            "PAXWWMD",
            "PAXSWMD",
            "PAXNWMD",
            "PAXUMD",
            "PAXQFD",
        ],
    )
    coerce_numeric(pax, pax.columns)
    pax["cycle"] = CYCLES[suffix]["label"]
    pax["is_full_protocol_day"] = pax["PAXDAYD"].between(2, 8)
    pax["wear_min_wake_sleep"] = pax["PAXWWMD"].fillna(0) + pax["PAXSWMD"].fillna(0)
    pax["valid_day_v2"] = (
        pax["is_full_protocol_day"]
        & (pax["wear_min_wake_sleep"] >= 1200)
        & (pax["PAXWWMD"] >= 600)
        & (pax["PAXVMD"] >= 1200)
    )
    pax["mims_per_valid_min"] = pax["PAXMTSD"] / pax["PAXVMD"].replace(0, np.nan)
    valid = pax[pax["valid_day_v2"]].copy()
    grouped = valid.groupby("SEQN", as_index=False).agg(
        pax_valid_days_v2=("valid_day_v2", "sum"),
        pax_mean_mims_per_valid_min=("mims_per_valid_min", "mean"),
        pax_mean_daily_mims=("PAXMTSD", "mean"),
        pax_mean_wake_wear_min=("PAXWWMD", "mean"),
        pax_mean_sleep_wear_min=("PAXSWMD", "mean"),
        pax_mean_nonwear_min=("PAXNWMD", "mean"),
        pax_mean_unknown_min=("PAXUMD", "mean"),
    )
    all_days = pax.groupby("SEQN", as_index=False).agg(
        pax_days_observed=("PAXDAYD", "nunique"),
        pax_full_days_observed=("is_full_protocol_day", "sum"),
    )
    out = all_days.merge(grouped, on="SEQN", how="left")
    out["pax_has_4_valid_days_v2"] = out["pax_valid_days_v2"].fillna(0) >= 4
    return out


def build_core_cycle(paths: Paths, suffix: str) -> pd.DataFrame:
    demo_cols = [
        "SEQN",
        "SDDSRVYR",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
        "WTMEC2YR",
        "SDMVSTRA",
        "SDMVPSU",
    ]
    demo = read_module(paths, suffix, "DEMO", demo_cols)
    coerce_numeric(demo, demo_cols)
    demo["cycle"] = CYCLES[suffix]["label"]
    demo["WTMEC4YR"] = demo["WTMEC2YR"] / 2.0
    demo["adult20"] = demo["RIDAGEYR"] >= 20
    demo["female"] = (demo["RIAGENDR"] == 2).astype(float)

    mcq_cols = [
        "SEQN",
        "MCQ160A",
        "MCQ195",
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
        "MCQ160G",
        "MCQ160K",
        "MCQ160O",
        "MCQ220",
    ]
    mcq = read_module(paths, suffix, "MCQ", None)
    needed_mcq = [c for c in mcq_cols if c in mcq.columns]
    mcq = mcq[needed_mcq].copy()
    coerce_numeric(mcq, mcq.columns)
    mcq["arthritis"] = valid_yes_no(mcq["MCQ160A"]) if "MCQ160A" in mcq.columns else np.nan
    mcq["arthritis_type_code"] = mcq["MCQ195"] if "MCQ195" in mcq.columns else np.nan
    cvd_cols = [c for c in ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"] if c in mcq.columns]
    mcq["cvd_history"] = np.where(mcq[cvd_cols].eq(1).any(axis=1), 1.0, np.where(mcq[cvd_cols].eq(2).any(axis=1), 0.0, np.nan)) if cvd_cols else np.nan
    resp_cols = [c for c in ["MCQ160G", "MCQ160K", "MCQ160O"] if c in mcq.columns]
    mcq["respiratory_history"] = np.where(mcq[resp_cols].eq(1).any(axis=1), 1.0, np.where(mcq[resp_cols].eq(2).any(axis=1), 0.0, np.nan)) if resp_cols else np.nan
    mcq["cancer_history"] = valid_yes_no(mcq["MCQ220"]) if "MCQ220" in mcq.columns else np.nan
    mcq = mcq[["SEQN", "arthritis", "arthritis_type_code", "cvd_history", "respiratory_history", "cancer_history"]]

    bmx = read_module(paths, suffix, "BMX", ["SEQN", "BMXBMI", "BMXWAIST"])
    coerce_numeric(bmx, bmx.columns)

    bpq = read_module(paths, suffix, "BPQ", None)
    bpq = bpq[[c for c in ["SEQN", "BPQ020", "BPQ050A"] if c in bpq.columns]].copy()
    coerce_numeric(bpq, bpq.columns)
    bpq["hypertension_self_report"] = np.where(
        bpq[[c for c in ["BPQ020", "BPQ050A"] if c in bpq.columns]].eq(1).any(axis=1),
        1.0,
        np.where(bpq[[c for c in ["BPQ020", "BPQ050A"] if c in bpq.columns]].eq(2).any(axis=1), 0.0, np.nan),
    )
    bpq = bpq[["SEQN", "hypertension_self_report"]]

    diq = read_module(paths, suffix, "DIQ", ["SEQN", "DIQ010"])
    coerce_numeric(diq, diq.columns)
    diq["diabetes_self_report"] = np.where(diq["DIQ010"] == 1, 1.0, np.where(diq["DIQ010"].isin([2, 3]), 0.0, np.nan))
    diq = diq[["SEQN", "diabetes_self_report"]]

    biopro = read_module(paths, suffix, "BIOPRO", ["SEQN", "LBXSCR"])
    coerce_numeric(biopro, biopro.columns)

    alb = read_module(paths, suffix, "ALB_CR", ["SEQN", "URDACT"])
    coerce_numeric(alb, alb.columns)

    smq = read_module(paths, suffix, "SMQ", ["SEQN", "SMQ020", "SMQ040"])
    coerce_numeric(smq, smq.columns)
    smq["smoking_status"] = np.nan
    smq.loc[smq["SMQ020"] == 2, "smoking_status"] = 0
    smq.loc[(smq["SMQ020"] == 1) & (smq["SMQ040"].isin([1, 2])), "smoking_status"] = 2
    smq.loc[(smq["SMQ020"] == 1) & (smq["SMQ040"] == 3), "smoking_status"] = 1
    smq = smq[["SEQN", "smoking_status"]]

    alq = read_module(paths, suffix, "ALQ", ["SEQN", "ALQ101", "ALQ110", "ALQ120Q"])
    coerce_numeric(alq, alq.columns)
    alq["any_alcohol_past_year"] = np.nan
    alq.loc[(alq.get("ALQ101") == 1) | (alq.get("ALQ110") == 1), "any_alcohol_past_year"] = 1
    alq.loc[(alq.get("ALQ101") == 2) | (alq.get("ALQ110") == 2), "any_alcohol_past_year"] = 0
    alq = alq[["SEQN", "any_alcohol_past_year"]]

    dpq = phq9_score(read_module(paths, suffix, "DPQ", None))
    slq = read_module(paths, suffix, "SLQ", ["SEQN", "SLD010H", "SLQ050", "SLQ060"])
    coerce_numeric(slq, slq.columns)
    slq = slq.rename(columns={"SLD010H": "self_report_sleep_hours", "SLQ050": "doctor_sleep_disorder", "SLQ060": "told_sleep_disorder"})

    pfq = pfq_score_v2(read_module(paths, suffix, "PFQ", None))
    pax = build_paxday_features(paths, suffix)

    frames = [demo, mcq, bmx, bpq, diq, biopro, alb, smq, alq, dpq, slq, pfq, pax]
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="SEQN", how="left")
    return out


def download_mortality(paths: Paths) -> None:
    paths.external_mortality.mkdir(parents=True, exist_ok=True)
    for filename, url in MORTALITY_URLS.items():
        dest = paths.external_mortality / filename
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {filename} ...", flush=True)
        urllib.request.urlretrieve(url, dest)


def parse_mortality_file(path: Path, suffix: str) -> pd.DataFrame:
    # Public-use 2019 linked mortality fixed-width positions:
    # SEQN=1-6, ELIGSTAT=15, MORTSTAT=16, UCOD_LEADING=17-19,
    # DIABETES=20, HYPERTEN=21, PERMTH_INT=43-45, PERMTH_EXM=46-48.
    colspecs = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
    names = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "lmf_diabetes", "lmf_hypertension", "PERMTH_INT", "PERMTH_EXM"]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str)
    for col in ["SEQN", "ELIGSTAT", "MORTSTAT", "lmf_diabetes", "lmf_hypertension", "PERMTH_INT", "PERMTH_EXM"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cycle"] = CYCLES[suffix]["label"]
    df["mortality_link_version"] = "2019_public"
    df["mortstat"] = np.where(df["MORTSTAT"] == 1, 1.0, np.where(df["MORTSTAT"] == 0, 0.0, np.nan))
    df["eligible_mortality"] = df["ELIGSTAT"] == 1
    ucod = df["UCOD_LEADING"].astype(str).str.zfill(3)
    df["cvd_death"] = np.where((df["mortstat"] == 1) & ucod.isin(["001", "005"]), 1.0, np.where(df["mortstat"].notna(), 0.0, np.nan))
    df["heart_death"] = np.where((df["mortstat"] == 1) & (ucod == "001"), 1.0, np.where(df["mortstat"].notna(), 0.0, np.nan))
    df["cerebrovascular_death"] = np.where((df["mortstat"] == 1) & (ucod == "005"), 1.0, np.where(df["mortstat"].notna(), 0.0, np.nan))
    df["cancer_death"] = np.where((df["mortstat"] == 1) & (ucod == "002"), 1.0, np.where(df["mortstat"].notna(), 0.0, np.nan))
    df["respiratory_death"] = np.where((df["mortstat"] == 1) & (ucod == "003"), 1.0, np.where(df["mortstat"].notna(), 0.0, np.nan))
    df["permth_exm"] = df["PERMTH_EXM"]
    return df[
        [
            "SEQN",
            "cycle",
            "mortality_link_version",
            "eligible_mortality",
            "mortstat",
            "cvd_death",
            "heart_death",
            "cerebrovascular_death",
            "cancer_death",
            "respiratory_death",
            "UCOD_LEADING",
            "permth_exm",
        ]
    ]


def build_mortality(paths: Paths) -> pd.DataFrame:
    download_mortality(paths)
    frames = []
    for suffix, cfg in CYCLES.items():
        frames.append(parse_mortality_file(paths.external_mortality / cfg["mort"], suffix))
    return pd.concat(frames, ignore_index=True)


def weighted_mean_sd(x: pd.Series, w: pd.Series) -> tuple[float, float]:
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    xv = x[mask].astype(float).to_numpy()
    wv = w[mask].astype(float).to_numpy()
    mu = np.average(xv, weights=wv)
    var = np.average((xv - mu) ** 2, weights=wv)
    return float(mu), float(math.sqrt(var))


def add_standardized_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["pfq_score_v2", "pax_mean_mims_per_valid_min", "BMXBMI", "RIDAGEYR", "phq9_score", "self_report_sleep_hours"]:
        if col not in df.columns:
            continue
        mu, sd = weighted_mean_sd(df[col], df["WTMEC4YR"])
        if pd.notna(sd) and sd > 0:
            df[f"{col}_z_wt"] = (df[col] - mu) / sd
        else:
            df[f"{col}_z_wt"] = np.nan
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    domains = {
        "all_rows": pd.Series(True, index=df.index),
        "adult20": df["adult20"] == True,
        "adult20_mortality_eligible": (df["adult20"] == True) & (df["eligible_mortality"] == True),
        "adult20_paxday_4_valid": (df["adult20"] == True) & (df["pax_has_4_valid_days_v2"] == True),
        "adult20_paxday_4_valid_mortality": (df["adult20"] == True) & (df["pax_has_4_valid_days_v2"] == True) & (df["eligible_mortality"] == True),
    }
    for name, mask in domains.items():
        sub = df[mask].copy()
        rows.append(
            {
                "domain": name,
                "n": len(sub),
                "weighted_n_mec4yr": sub["WTMEC4YR"].sum(skipna=True),
                "arthritis_n": int(sub["arthritis"].eq(1).sum()) if "arthritis" in sub else np.nan,
                "mortality_events": int(sub["mortstat"].eq(1).sum()) if "mortstat" in sub else np.nan,
                "cvd_deaths": int(sub["cvd_death"].eq(1).sum()) if "cvd_death" in sub else np.nan,
                "pfq_score_v2_nonmissing": int(sub["pfq_score_v2"].notna().sum()) if "pfq_score_v2" in sub else np.nan,
                "pax_valid_days_median": sub["pax_valid_days_v2"].median(skipna=True) if "pax_valid_days_v2" in sub else np.nan,
                "mean_followup_months": sub["permth_exm"].mean(skipna=True) if "permth_exm" in sub else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_codebook(paths: Paths, df: pd.DataFrame) -> None:
    lines = [
        "# T1 analysis base table codebook",
        "",
        "Generated by `scripts/01_build_base_table.py`.",
        "",
        "Important design decisions:",
        "",
        "- Mortality file: public-use 2019 linked mortality file. The 2022 follow-up is restricted-use and requires NCHS RDC access.",
        "- Survey weights: `WTMEC4YR = WTMEC2YR / 2` for the two pooled NHANES cycles.",
        "- PFQ v2: PFQ061A-T positive codes are 2/3/4; 7 and 9 are missing; code 5 (`do not do this activity`) is not counted as a positive limitation in the primary score and is tracked in `pfq061_do_not_do_count`. PFQ061 structural skips after PFQ059=2 are coded as 0 limitations.",
        "- PFQ v2 broad items: PFQ049, PFQ051, PFQ057, PFQ059; code 1 is positive, code 2 is negative, 7/9 are missing.",
        "- PAXDAY valid day v2: day 2-8, wake+sleep wear minutes >=1200, wake wear minutes >=600, valid minutes >=1200.",
        "- CVD mortality definition: UCOD_LEADING 001 (heart disease) or 005 (cerebrovascular disease). Heart and cerebrovascular deaths are also kept separately.",
        "",
        "Columns:",
        "",
    ]
    for col in df.columns:
        lines.append(f"- `{col}`")
    (paths.docs / "base_table_codebook.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhanes-root", default="nhanes_raw", type=Path)
    parser.add_argument("--out-dir", default="T1_RAR_arthritis_mortality_supplemental", type=Path)
    args = parser.parse_args()

    paths = Paths(args.nhanes_root, args.out_dir)
    for p in [paths.data, paths.results, paths.docs, paths.external_mortality]:
        p.mkdir(parents=True, exist_ok=True)

    frames = []
    for suffix in CYCLES:
        print(f"Building {CYCLES[suffix]['label']} base table ...", flush=True)
        frames.append(build_core_cycle(paths, suffix))
    base = pd.concat(frames, ignore_index=True)
    mortality = build_mortality(paths)
    base = base.merge(mortality, on=["SEQN", "cycle"], how="left")
    base = add_standardized_columns(base)

    csv_path = paths.data / "t1_analysis_base_day_pfq_mortality.csv"
    parquet_path = paths.data / "t1_analysis_base_day_pfq_mortality.parquet"
    base.to_csv(csv_path, index=False)
    base.to_parquet(parquet_path, index=False)

    summary = summarize(base)
    summary.to_csv(paths.results / "base_feasibility_summary.csv", index=False)
    write_codebook(paths, base)

    manifest = {
        "script": "01_build_base_table.py",
        "rows": int(len(base)),
        "columns": int(base.shape[1]),
        "mortality_link_version": "2019_public",
        "note_2022_lmf": "The 2022 NHANES linked mortality follow-up is restricted-use and requires NCHS RDC access; no public 2022 cycle-specific files were available from the public FTP path.",
        "outputs": [str(csv_path), str(parquet_path), str(paths.results / "base_feasibility_summary.csv")],
    }
    (paths.results / "base_build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
