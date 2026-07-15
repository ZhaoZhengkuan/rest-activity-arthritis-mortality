# Rest-activity rhythm fragmentation, arthritis, and mortality (NHANES 2011–2014)

Analysis code for the study 

The pipeline derives non-parametric rest-activity rhythm (RAR) features from
NHANES wrist-accelerometry, links them to arthritis status, a Physical
Functioning (PFQ) functional-limitation score, and public-use linked mortality,
and runs the weighted cross-sectional, prognostic (Cox), and secondary
interventional-decomposition analyses reported in the manuscript.

> **Analytic scope.** The reported Python regression models use pooled MEC
> examination weights and robust variance/standard errors clustered on
> `SDMVPSU`. They do not incorporate `SDMVSTRA` in the model-based variance
> estimators and therefore are not full design-based complex-survey regression
> analyses. `SDMVSTRA` is used only in the PSU-within-stratum bootstrap in
> script `06`. The R/CMAverse file is a compatibility template and was not used
> to generate the reported results. No `survey::svycoxph` or CMAverse
> confirmation is claimed.

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
    ├── 03_weighted_gformula_mediation.py # MEC-weighted fixed-horizon g-formula (secondary analysis)
    ├── 04_weighted_cox_sensitivity.py    # weighted Cox: lag-gradient, cause-specific, permutation checks
    ├── 05_cmaverse_r_compatibility_template.R  # unused R/CMAverse compatibility template
    ├── 06_python_design_bootstrap.py     # 100-replicate PSU-within-strata bootstrap
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

**R** (only if independently using the unexecuted compatibility template in script `05`):

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

# 3. MEC-weighted fixed-horizon interventional decomposition (secondary analysis)
python code/03_weighted_gformula_mediation.py

# 4. Weighted Cox sensitivity: lag-gradient exclusions, cause-specific, permutation
python code/04_weighted_cox_sensitivity.py

# 5. (Not used for reported results) R/CMAverse compatibility template
Rscript code/05_cmaverse_r_compatibility_template.R

# 6. PSU-within-strata bootstrap intervals for 03/04 (default 100 replicates)
python code/06_python_design_bootstrap.py

# 7. Cross-sectional arthritis → RAR feature associations (Table 2)
python code/07_arm1_weighted_arthritis_rar.py
```

Each script exposes `--help`. Common options include `--out-dir` (working
directory), `--tau` (mediation horizon, default 60 months), and `--n-boot`
(bootstrap replicates).

---

## Methods notes

- **Weighting and variance.** Estimates use the pooled 4-year MEC examination
  weight (`WTMEC4YR = WTMEC2YR / 2`). Weighted least-squares and Cox models use
  robust standard errors/variance clustered on `SDMVPSU`. `SDMVSTRA` is not
  included in those model-based variance estimators. These are weighted,
  PSU-clustered models, not full design-based complex-survey regression models.
- **Primary exposure.** Intradaily variability standardised to an examination-weighted
  z-score (`iv_z_wt`).
- **Valid-day rule.** Days 2–8 with ≥1200 wear minutes (≥600 wake-wear, ≥1200
  valid minutes); participants with ≥4 valid days are retained.
- **Secondary interventional decomposition.** Functional limitation (PFQ) is
  treated as a post-exposure confounder. The Python fixed-horizon g-formula
  (`03`) reports weighted and unweighted final standardisations, with uncertainty
  from 100 PSU-within-stratum bootstrap replicates. Script `05` is an unused
  compatibility template and is not evidence of an R/CMAverse confirmation.
- **Interpretation.** Cause-specific and mediation outputs are secondary or
  exploratory. The repository does not establish causality, prediction
  performance, etiologic specificity, or full complex-survey inference.

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
