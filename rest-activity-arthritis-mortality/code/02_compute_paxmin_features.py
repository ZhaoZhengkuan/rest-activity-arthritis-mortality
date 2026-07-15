#!/usr/bin/env python3
"""
Compute minute-level rest-activity rhythm and sleep-regularity features from
NHANES 2011-2014 PAXMIN files.

Primary locked mediator for the mediation package:
    iv_z_wt

Secondary/exploratory features:
    IS, RA, M10/L5 timing, SRI, wake/sleep transition calibration, and an
    active/inactive transition-fragmentation metric.

The script streams PAXMIN with pyreadstat and only keeps the columns needed for
the planned features. It uses the same valid-day definition as
01_build_base_table.py:
    day 2-8, wake+sleep wear minutes >=1200, wake wear minutes >=600,
    valid minutes >=1200.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyreadstat


CYCLES = {
    "G": {"cycle": "2011_2012", "label": "2011-2012"},
    "H": {"cycle": "2013_2014", "label": "2013-2014"},
}

PAXMIN_COLS = ["SEQN", "PAXDAYM", "PAXSSNMP", "PAXMTSM", "PAXPREDM", "PAXQFM"]
MINUTES_PER_DAY = 1440


@dataclass
class Paths:
    nhanes_root: Path
    out_dir: Path

    @property
    def raw(self) -> Path:
        return self.nhanes_root / "raw"

    @property
    def data(self) -> Path:
        return self.out_dir / "data"

    @property
    def results(self) -> Path:
        return self.out_dir / "results"

    @property
    def logs(self) -> Path:
        return self.out_dir / "logs"


def read_xpt(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    df, _ = pyreadstat.read_xport(str(path), usecols=list(columns) if columns else None)
    return df


def valid_day_map(paths: Paths, suffix: str) -> dict[int, set[int]]:
    path = paths.raw / CYCLES[suffix]["cycle"] / f"PAXDAY_{suffix}.XPT"
    cols = ["SEQN", "PAXDAYD", "PAXVMD", "PAXWWMD", "PAXSWMD"]
    df = read_xpt(path, cols)
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wear_min_wake_sleep"] = df["PAXWWMD"].fillna(0) + df["PAXSWMD"].fillna(0)
    ok = (
        df["PAXDAYD"].between(2, 8)
        & (df["wear_min_wake_sleep"] >= 1200)
        & (df["PAXWWMD"] >= 600)
        & (df["PAXVMD"] >= 1200)
    )
    keep = defaultdict(set)
    for seqn, day in df.loc[ok, ["SEQN", "PAXDAYD"]].itertuples(index=False):
        keep[int(seqn)].add(int(day))
    return dict(keep)


def circular_window_extrema(profile: np.ndarray, window: int, mode: str, min_fraction: float = 0.8) -> tuple[float, int]:
    valid = np.isfinite(profile)
    values = np.where(valid, profile, 0.0)
    doubled_values = np.concatenate([values, values])
    doubled_valid = np.concatenate([valid.astype(float), valid.astype(float)])
    kernel = np.ones(window)
    sums = np.convolve(doubled_values, kernel, mode="valid")[:MINUTES_PER_DAY]
    counts = np.convolve(doubled_valid, kernel, mode="valid")[:MINUTES_PER_DAY]
    means = np.where(counts >= window * min_fraction, sums / counts, np.nan)
    if np.all(np.isnan(means)):
        return np.nan, -1
    if mode == "max":
        idx = int(np.nanargmax(means))
    else:
        idx = int(np.nanargmin(means))
    return float(means[idx]), idx


def _as_arrays(rows: pd.DataFrame | list[tuple[int, int, float, int, float]]):
    if isinstance(rows, pd.DataFrame):
        return (
            rows["day"].to_numpy(dtype="int64", copy=False),
            rows["minute"].to_numpy(dtype="int64", copy=False),
            rows["mims"].to_numpy(dtype="float64", copy=False),
            rows["pred"].to_numpy(dtype="int64", copy=False),
            rows["qf"].to_numpy(dtype="float64", copy=False),
        )
    arr = np.asarray(rows, dtype="float64")
    if arr.size == 0:
        return None
    return (
        arr[:, 0].astype("int64"),
        arr[:, 1].astype("int64"),
        arr[:, 2].astype("float64"),
        arr[:, 3].astype("int64"),
        arr[:, 4].astype("float64"),
    )


def compute_features_for_person(seqn: int, rows: pd.DataFrame | list[tuple[int, int, float, int, float]], min_valid_days: int = 4) -> dict:
    arrays = _as_arrays(rows)
    if arrays is None:
        return {"SEQN": seqn, "feature_status": "no_rows"}
    day_arr, minute_arr, mims_arr, pred_arr, qf_arr = arrays

    days = sorted(set(day_arr.tolist()))
    if len(days) < min_valid_days:
        return {"SEQN": seqn, "feature_status": "lt_min_valid_days", "rar_valid_days": len(days)}

    day_index = {d: i for i, d in enumerate(days)}
    x = np.full((len(days), MINUTES_PER_DAY), np.nan, dtype="float64")
    state = np.full((len(days), MINUTES_PER_DAY), np.nan, dtype="float64")
    wake_x = np.full((len(days), MINUTES_PER_DAY), np.nan, dtype="float64")

    day_pos = pd.Series(day_arr).map(day_index).to_numpy()
    base_ok = (
        pd.notna(day_pos)
        & (minute_arr >= 0)
        & (minute_arr < MINUTES_PER_DAY)
        & np.isfinite(mims_arr)
        & (mims_arr != -0.01)
        & np.isfinite(qf_arr)
        & (qf_arr <= 0)
    )
    day_pos = day_pos.astype("float64")
    didx = day_pos[base_ok].astype("int64")
    midx = minute_arr[base_ok].astype("int64")
    mvals = np.maximum(mims_arr[base_ok].astype("float64"), 0.0)
    pvals = pred_arr[base_ok].astype("int64")

    wear = np.isin(pvals, [1, 2])
    x[didx[wear], midx[wear]] = mvals[wear]
    wake = pvals == 1
    sleep = pvals == 2
    state[didx[wake], midx[wake]] = 1.0
    wake_x[didx[wake], midx[wake]] = mvals[wake]
    state[didx[sleep], midx[sleep]] = 0.0

    valid_day_counts = np.isfinite(x).sum(axis=1)
    retained = valid_day_counts >= 1200
    if retained.sum() < min_valid_days:
        return {
            "SEQN": seqn,
            "feature_status": "lt_min_valid_days_after_minute_qc",
            "rar_valid_days": int(retained.sum()),
        }
    x = x[retained]
    state = state[retained]
    wake_x = wake_x[retained]
    days_retained = np.array(days)[retained]

    flat = x.reshape(-1)
    finite = np.isfinite(flat)
    n = int(finite.sum())
    if n < 2:
        return {"SEQN": seqn, "feature_status": "insufficient_valid_minutes", "rar_valid_days": int(len(days_retained))}
    mean_x = float(np.nanmean(flat))
    denom = float(np.nansum((flat[finite] - mean_x) ** 2))

    both = np.isfinite(flat[1:]) & np.isfinite(flat[:-1])
    diff_num = float(np.sum((flat[1:][both] - flat[:-1][both]) ** 2))
    if denom > 0 and n > 1:
        iv = n * diff_num / ((n - 1) * denom)
    else:
        iv = np.nan

    profile = np.nanmean(x, axis=0)
    profile_valid = np.isfinite(profile)
    if denom > 0 and profile_valid.sum() >= 720:
        is_value = n * float(np.nansum((profile[profile_valid] - mean_x) ** 2)) / (MINUTES_PER_DAY * denom)
    else:
        is_value = np.nan

    m10, m10_start = circular_window_extrema(profile, 600, "max")
    l5, l5_start = circular_window_extrema(profile, 300, "min")
    if np.isfinite(m10) and np.isfinite(l5) and (m10 + l5) > 0:
        ra = (m10 - l5) / (m10 + l5)
    else:
        ra = np.nan

    if state.shape[0] >= 2:
        same = []
        valid_pairs = []
        for i in range(state.shape[0] - 1):
            ok = np.isfinite(state[i]) & np.isfinite(state[i + 1])
            if ok.any():
                same.append(np.sum(state[i][ok] == state[i + 1][ok]))
                valid_pairs.append(np.sum(ok))
        if valid_pairs and sum(valid_pairs) > 0:
            sri = -100.0 + 200.0 * (sum(same) / sum(valid_pairs))
            sri_pairs = int(sum(valid_pairs))
        else:
            sri = np.nan
            sri_pairs = 0
    else:
        sri = np.nan
        sri_pairs = 0

    wx = wake_x.reshape(-1)
    wf = np.isfinite(wx)
    wake_minutes = int(wf.sum())
    if wake_minutes > 1:
        # Secondary calibration metric. This is not the locked mediator.
        active = wx > 1.0
        pair_ok = np.isfinite(wx[1:]) & np.isfinite(wx[:-1])
        transitions = int(np.sum(active[1:][pair_ok] != active[:-1][pair_ok]))
        transition_fragmentation = transitions / (pair_ok.sum() / 60.0) if pair_ok.sum() > 0 else np.nan
        wake_mims_mean = float(np.nanmean(wx))
        wake_mims_p50 = float(np.nanmedian(wx))
    else:
        transitions = 0
        transition_fragmentation = np.nan
        wake_mims_mean = np.nan
        wake_mims_p50 = np.nan

    return {
        "SEQN": seqn,
        "feature_status": "ok",
        "rar_valid_days": int(len(days_retained)),
        "rar_valid_minutes": n,
        "iv": iv,
        "is": is_value,
        "ra": ra,
        "m10": m10,
        "l5": l5,
        "m10_start_minute": m10_start,
        "l5_start_minute": l5_start,
        "sri": sri,
        "sri_valid_pairs": sri_pairs,
        "wake_minutes": wake_minutes,
        "wake_mims_mean": wake_mims_mean,
        "wake_mims_p50": wake_mims_p50,
        "transition_fragmentation_mims_gt1_per_wake_hour": transition_fragmentation,
        "active_inactive_transitions_mims_gt1": transitions,
    }


def weighted_mean_sd(x: pd.Series, w: pd.Series) -> tuple[float, float]:
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    xv = x[mask].astype(float).to_numpy()
    wv = w[mask].astype(float).to_numpy()
    mu = np.average(xv, weights=wv)
    sd = math.sqrt(np.average((xv - mu) ** 2, weights=wv))
    return float(mu), float(sd)


def add_z_scores(features: pd.DataFrame, base_path: Path | None) -> pd.DataFrame:
    if base_path is None or not base_path.exists():
        for col in ["iv", "is", "ra", "sri", "transition_fragmentation_mims_gt1_per_wake_hour"]:
            if col in features.columns:
                mu = features[col].mean(skipna=True)
                sd = features[col].std(skipna=True)
                features[f"{col}_z"] = (features[col] - mu) / sd if sd and sd > 0 else np.nan
        return features
    base = pd.read_parquet(base_path) if base_path.suffix == ".parquet" else pd.read_csv(base_path)
    keep = base[
        (base["adult20"] == True)
        & (base["eligible_mortality"] == True)
        & (base["pax_has_4_valid_days_v2"] == True)
    ][["SEQN", "WTMEC4YR"]]
    merged = features.merge(keep, on="SEQN", how="left")
    for col in ["iv", "is", "ra", "sri", "transition_fragmentation_mims_gt1_per_wake_hour"]:
        if col not in merged.columns:
            continue
        mu, sd = weighted_mean_sd(merged[col], merged["WTMEC4YR"])
        if pd.notna(sd) and sd > 0:
            features[f"{col}_z_wt"] = (features[col] - mu) / sd
        else:
            features[f"{col}_z_wt"] = np.nan
    # Locked primary mediator alias.
    if "iv_z_wt" in features.columns:
        features["iv_z"] = features["iv_z_wt"]
    return features


def _decode_byte_codes(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(series.str.decode("utf-8", errors="ignore"), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def iter_paxmin_chunks(path: Path, chunk_size: int):
    reader = pd.read_sas(str(path), format="xport", chunksize=chunk_size)
    for chunk in reader:
        chunk = chunk[PAXMIN_COLS].copy()
        chunk["SEQN"] = pd.to_numeric(chunk["SEQN"], errors="coerce")
        chunk["PAXDAYM"] = _decode_byte_codes(chunk["PAXDAYM"])
        chunk["PAXPREDM"] = _decode_byte_codes(chunk["PAXPREDM"])
        for col in ["PAXSSNMP", "PAXMTSM", "PAXQFM"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            # pandas.read_sas represents XPORT zeros in some integer-like
            # fields as tiny positive floats around 5e-79. They are zeros for
            # our purposes.
            chunk.loc[chunk[col].abs() < 1e-30, col] = 0.0
        chunk = chunk.dropna(subset=["SEQN", "PAXDAYM", "PAXPREDM"])
        yield chunk


def process_cycle(paths: Paths, suffix: str, chunk_size: int, limit_chunks: int | None = None) -> pd.DataFrame:
    valid_map = valid_day_map(paths, suffix)
    eligible_seqns = set(valid_map.keys())
    path = paths.raw / CYCLES[suffix]["cycle"] / f"PAXMIN_{suffix}.XPT"
    start = time.time()
    results = []
    current_seqn = None
    current_parts: list[pd.DataFrame] = []
    chunks_read = 0
    rows_seen = 0
    rows_kept = 0

    def flush_current() -> None:
        nonlocal current_seqn, current_parts, results
        if current_seqn is not None:
            frame = pd.concat(current_parts, ignore_index=True) if current_parts else pd.DataFrame(columns=["day", "minute", "mims", "pred", "qf"])
            res = compute_features_for_person(current_seqn, frame)
            res["cycle"] = CYCLES[suffix]["label"]
            results.append(res)
        current_parts = []

    for chunk in iter_paxmin_chunks(path, chunk_size):
        chunks_read += 1
        rows_seen += len(chunk)
        chunk = chunk[chunk["SEQN"].isin(eligible_seqns)].copy()
        if chunk.empty:
            if limit_chunks and chunks_read >= limit_chunks:
                break
            continue
        chunk["PAXDAYM_INT"] = chunk["PAXDAYM"].astype("int64")
        chunk["SEQN_INT"] = chunk["SEQN"].astype("int64")
        day_ok = np.fromiter(
            (
                int(seqn) in valid_map and int(day) in valid_map[int(seqn)]
                for seqn, day in zip(chunk["SEQN_INT"].to_numpy(), chunk["PAXDAYM_INT"].to_numpy())
            ),
            dtype=bool,
            count=len(chunk),
        )
        chunk = chunk.loc[day_ok]
        rows_kept += len(chunk)
        if not chunk.empty:
            minute = np.rint(chunk["PAXSSNMP"].to_numpy(dtype=float) / 4800.0).astype("int64") % MINUTES_PER_DAY
            chunk = chunk.assign(minute_of_day=minute)
            compact = chunk.rename(
                columns={
                    "PAXDAYM_INT": "day",
                    "minute_of_day": "minute",
                    "PAXMTSM": "mims",
                    "PAXPREDM": "pred",
                    "PAXQFM": "qf",
                }
            )[["SEQN_INT", "day", "minute", "mims", "pred", "qf"]]
            for seqn, group in compact.groupby("SEQN_INT", sort=False):
                seqn = int(seqn)
                if current_seqn is None:
                    current_seqn = seqn
                if seqn != current_seqn:
                    flush_current()
                    current_seqn = seqn
                current_parts.append(group.drop(columns=["SEQN_INT"]).copy())
        if chunks_read % 10 == 0:
            elapsed = time.time() - start
            print(
                f"{CYCLES[suffix]['label']} chunk={chunks_read} rows_seen={rows_seen:,} rows_kept={rows_kept:,} features={len(results):,} elapsed_min={elapsed/60:.1f}",
                flush=True,
            )
        if limit_chunks and chunks_read >= limit_chunks:
            break
    flush_current()
    out = pd.DataFrame(results)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhanes-root", default="nhanes_raw", type=Path)
    parser.add_argument("--out-dir", default="T1_RAR_arthritis_mortality_supplemental", type=Path)
    parser.add_argument("--chunk-size", default=1_000_000, type=int)
    parser.add_argument("--limit-chunks", default=None, type=int)
    parser.add_argument("--base-table", default=None, type=Path)
    args = parser.parse_args()

    paths = Paths(args.nhanes_root, args.out_dir)
    for p in [paths.data, paths.results, paths.logs]:
        p.mkdir(parents=True, exist_ok=True)
    base_path = args.base_table or (paths.data / "t1_analysis_base_day_pfq_mortality.parquet")

    all_features = []
    cycle_summaries = []
    for suffix in CYCLES:
        cycle_features = process_cycle(paths, suffix, args.chunk_size, args.limit_chunks)
        all_features.append(cycle_features)
        cycle_summaries.append(
            {
                "cycle": CYCLES[suffix]["label"],
                "n_features": int(len(cycle_features)),
                "n_ok": int(cycle_features.get("feature_status", pd.Series(dtype=str)).eq("ok").sum()),
            }
        )
        cycle_features.to_csv(paths.data / f"paxmin_rar_features_{suffix}.csv", index=False)

    features = pd.concat(all_features, ignore_index=True)
    features = add_z_scores(features, base_path)
    out_csv = paths.data / "paxmin_rar_features_2011_2014.csv"
    out_parquet = paths.data / "paxmin_rar_features_2011_2014.parquet"
    features.to_csv(out_csv, index=False)
    features.to_parquet(out_parquet, index=False)

    summary_rows = []
    for status, sub in features.groupby("feature_status", dropna=False):
        row = {"feature_status": status, "n": len(sub)}
        for col in ["iv", "iv_z_wt", "is", "ra", "sri", "transition_fragmentation_mims_gt1_per_wake_hour"]:
            if col in sub.columns:
                row[f"{col}_nonmissing"] = int(sub[col].notna().sum())
                row[f"{col}_mean"] = sub[col].mean(skipna=True)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths.results / "paxmin_feature_summary.csv", index=False)
    (paths.results / "paxmin_feature_manifest.json").write_text(
        json.dumps(
            {
                "script": "02_compute_paxmin_features.py",
                "locked_primary_mediator": "iv_z_wt (also copied to iv_z)",
                "chunk_size": args.chunk_size,
                "limit_chunks": args.limit_chunks,
                "cycle_summaries": cycle_summaries,
                "outputs": [str(out_csv), str(out_parquet), str(paths.results / "paxmin_feature_summary.csv")],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
