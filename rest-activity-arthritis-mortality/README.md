# Rest-activity rhythm fragmentation, arthritis, and mortality (NHANES 2011–2014)

Analysis code for the study **"Objectively measured rest-activity rhythm
fragmentation in adults with arthritis: associations with functional limitation
and mortality in a nationally representative cohort."**

The pipeline derives non-parametric rest-activity rhythm (RAR) features from
NHANES wrist-accelerometry, links them to arthritis status, a Physical
Functioning (PFQ) functional-limitation score, and public-use linked mortality,
and runs the survey-weighted cross-sectional, prognostic (Cox), and
interventional-mediation analyses reported in the manuscript.

> **This repository contains code only.** No NHANES data, derived analytic
> tables, or results are tracked here (see [`.gitignore`](.gitignore)). All
> inputs are publicly available from NCHS — see **Data** below.

---

## Data

The analysis uses only publicly available data from the US National Center for
Health Statistics (NCHS); nothing proprietary is required.

- **NHANES 2011–2012 and 2013–2014** questionnaire, examination, and
  minute-level accelerometry (PAXMIN) files:
  https://wwwn.cdc.gov/nchs/nhanes/
- **Public-use Linked Mortality Files (2019)**:
  https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

Download the required NHANES SAS transport (`.XPT`) files and place them under a
local directory (default expected name: `nhanes_raw/`).
Script `01` will additionally fetch the public mortality files automatically as
a fallback. Raw data files are `.gitignore`d and must **not** be committed.

---

## Repository structure

```
.
├── README.md
├── LICENSE                 # MIT (placeholder — change to your preferred licence)
├── requirements.txt        # Python dependencies
├── .gitignore              # excludes all data, derived tables, and results
└── code/
    ├── 01_build_base_table.py            # person-level base table (design, arthritis, PFQ, covariates, mortality)
    ├── 02_compute_paxmin_features.py     # minute-level RAR features (IV, IS, RA, M10/L5, SRI, ...)
    ├── 03_weighted_gformula_mediation.py # survey-weighted interventional g-formula (60-month mediation)
    ├── 04_weighted_cox_sensitivity.py    # weighted Cox: lag-gradient, cause-specific, permutation checks
    ├── 05_cmaverse_r_compatibility_template.R  # R/CMAverse interventional-survival confirmation (template)
    ├── 06_python_design_bootstrap.py     # PSU-within-strata design bootstrap for 03/04
    └── 07_arm1_weighted_arthritis_rar.py # cross-sectional arthritis → RAR feature regressions (Table 2)
```

Running the scripts creates a working directory (default
`T1_RAR_arthritis_mortality_supplemental/`) with `data/` (derived tables) and
`results/` / `docs/` (analysis outputs). That directory is `.gitignore`d.

---

## Setup

**Python** (tested on 3.12):

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

**R** (only for the optional confirmation in script `05`):

```r
install.packages(c("readr", "dplyr", "survival", "survey", "CMAverse"))
```

---

## Running the pipeline

Run from the repository root. Steps `01`–`02` build the analytic tables from raw
NHANES; steps `03`–`07` consume those tables.

```bash
# 1. Person-level base table (design vars, arthritis, PFQ score, covariates, mortality)
python code/01_build_base_table.py --nhanes-root nhanes_raw

# 2. Minute-level rest-activity rhythm features (locked primary exposure: iv_z_wt)
python code/02_compute_paxmin_features.py --nhanes-root nhanes_raw

# 3. Survey-weighted interventional g-formula mediation (60-month horizon)
python code/03_weighted_gformula_mediation.py

# 4. Weighted Cox sensitivity: lag-gradient exclusions, cause-specific, permutation
python code/04_weighted_cox_sensitivity.py

# 5. (Optional, in R) CMAverse interventional-survival confirmation
Rscript code/05_cmaverse_r_compatibility_template.R

# 6. Design-based (PSU-within-strata) bootstrap intervals for 03/04
python code/06_python_design_bootstrap.py

# 7. Cross-sectional arthritis → RAR feature associations (Table 2)
python code/07_arm1_weighted_arthritis_rar.py
```

Each script exposes `--help`. Common options include `--out-dir` (working
directory), `--tau` (mediation horizon, default 60 months), and `--n-boot`
(bootstrap replicates).

---

## Methods notes

- **Survey design.** All estimates use the pooled 4-year MEC examination weight
  (`WTMEC4YR = WTMEC2YR / 2`) with masked stratum (`SDMVSTRA`) and PSU
  (`SDMVPSU`) identifiers. The full-sample design is specified before
  restricting to the analytic subpopulation.
- **Primary exposure.** Intradaily variability standardised to a survey-weighted
  z-score (`iv_z_wt`).
- **Valid-day rule.** Days 2–8 with ≥1200 wear minutes (≥600 wake-wear, ≥1200
  valid minutes); participants with ≥4 valid days are retained.
- **Mediation.** Functional limitation (PFQ) is treated as an exposure-induced
  mediator–outcome confounder; the Python g-formula (`03`) reports both weighted
  and unweighted final standardisations as an audit, and `05` is provided so the
  preferred interventional-survival analysis can be confirmed in R/CMAverse.
- The Python scripts implement survey-weighted models with PSU-clustered robust
  variance as an executable substitute where an R `survey`/`CMAverse`
  environment is not available locally.

---

## Reproducibility

The pipeline is deterministic given the same NHANES inputs; stochastic steps
(bootstrap, permutation) are seeded (e.g. `--seed`, default `20260619`). Pin the
dependency versions in `requirements.txt` for bit-for-bit reproduction.

## Licence

Released under the MIT Licence (see [`LICENSE`](LICENSE)). The `LICENSE` file is
a placeholder default — replace it if you prefer a different licence.

## Citation

If you use this code, please cite the associated article (details to be added on
publication).
