#!/usr/bin/env Rscript

# R/CMAverse compatibility template for the T1 RAR/arthritis/mortality project.
#
# This script was not executed and did not generate any result reported in the
# manuscript. It is retained only as a compatibility template for independent
# future work. No R/CMAverse or full complex-survey confirmation is claimed.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(survival)
  library(survey)
  library(CMAverse)
})

out_dir <- "T1_RAR_arthritis_mortality_supplemental"
base <- read_csv(file.path(out_dir, "data", "t1_analysis_base_day_pfq_mortality.csv"), show_col_types = FALSE)
rar <- read_csv(file.path(out_dir, "data", "paxmin_rar_features_2011_2014.csv"), show_col_types = FALSE)

dat <- base %>%
  inner_join(rar, by = c("SEQN", "cycle")) %>%
  filter(
    adult20 == TRUE,
    eligible_mortality == TRUE,
    pax_has_4_valid_days_v2 == TRUE,
    feature_status == "ok"
  ) %>%
  mutate(
    arthritis = as.integer(arthritis),
    mortstat = as.integer(mortstat),
    WTMEC4YR = WTMEC2YR / 2
  ) %>%
  select(
    SEQN, permth_exm, mortstat, arthritis, iv_z_wt, pfq_score_v2_z_wt,
    RIDAGEYR_z_wt, female, RIDRETH3, DMDEDUC2, INDFMPIR, BMXBMI_z_wt,
    smoking_status, hypertension_self_report, diabetes_self_report,
    cvd_history, phq9_score_z_wt, self_report_sleep_hours_z_wt,
    WTMEC4YR, SDMVSTRA, SDMVPSU
  ) %>%
  tidyr::drop_na()

covars <- c(
  "RIDAGEYR_z_wt", "female", "factor(RIDRETH3)", "factor(DMDEDUC2)",
  "INDFMPIR", "BMXBMI_z_wt", "factor(smoking_status)",
  "hypertension_self_report", "diabetes_self_report", "cvd_history",
  "phq9_score_z_wt", "self_report_sleep_hours_z_wt"
)

# Compatibility check 1: independent weighted Cox. This mirrors the outcome
# model CMAverse fits internally on the gformula path and is used only to
# confirm the weighted hazard ratio for iv_z_wt. Efron ties are used to match
# the Python lifelines main analysis (lifelines default is Efron).
cox_formula <- as.formula(paste(
  "Surv(permth_exm, mortstat) ~ arthritis * iv_z_wt + pfq_score_v2_z_wt +",
  paste(covars, collapse = " + ")
))

cox_weighted <- coxph(
  cox_formula,
  data = dat,
  weights = WTMEC4YR,
  robust = TRUE,
  cluster = SDMVPSU,
  ties = "efron"
)
print(summary(cox_weighted))

# Preferred CMAverse call: interventional (g-formula) effects with a single
# locked mediator (iv_z_wt) and PFQ as a post-exposure confounder.
#
# Survival-outcome calling convention (important):
#   - yreg must be the CHARACTER name "coxph"; CMAverse then builds the Cox
#     model internally as
#       Surv(<outcome>, <event>) ~ arthritis + iv_z_wt + arthritis:iv_z_wt + basec
#     (the exposure-mediator interaction is added automatically by EMint = TRUE).
#   - outcome must be the TIME variable (permth_exm), and event the status
#     indicator (mortstat). Do NOT pass a pre-built Surv() formula as yreg and do
#     NOT set outcome = "mortstat"; that omits the survival time and is a
#     specification error for a time-to-event outcome.
#   - postc (pfq_score_v2_z_wt) is a post-exposure confounder of the
#     mediator-outcome relationship; only model = "gformula" (or "msm") supports
#     a non-empty postc, which is why interventional effects are estimated here.
#
# MEC survey weights are passed via weights = "WTMEC4YR". Per the CMAverse
# documentation, when a regression is fitted with prior weights the final
# weights are the product of the prior weights and the weights cmest computes
# internally, so the survey weights propagate through the component models.
res_cox <- cmest(
  data = dat,
  model = "gformula",
  outcome = "permth_exm",
  event = "mortstat",
  exposure = "arthritis",
  mediator = "iv_z_wt",
  basec = c(
    "RIDAGEYR_z_wt", "female", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
    "BMXBMI_z_wt", "smoking_status", "hypertension_self_report",
    "diabetes_self_report", "cvd_history", "phq9_score_z_wt",
    "self_report_sleep_hours_z_wt"
  ),
  postc = "pfq_score_v2_z_wt",
  mreg = list("linear"),
  postcreg = list("linear"),
  yreg = "coxph",
  astar = 0,
  a = 1,
  mval = list(NULL),
  EMint = TRUE,
  estimation = "imputation",
  inference = "bootstrap",
  nboot = 1000,
  boot.ci.type = "per",
  weights = "WTMEC4YR"
)

print(summary(res_cox))

# Notes for the final analysis:
# 1. Survey-design bootstrap. CMAverse's own bootstrap resamples rows as a simple
#    random sample and ignores SDMVSTRA/SDMVPSU, so its intervals can be too
#    narrow. For design-consistent CIs, resample PSUs within strata (matching the
#    Python design bootstrap in 06_python_design_bootstrap.py) and treat the
#    naive CMAverse interval as a comparison only.
# 2. Marginal standardization. The imputation g-formula averages counterfactual
#    risks over the analysis sample. Compare the cmest point estimates with a
#    manual MEC-weighted marginal standardization to confirm the target
#    population; report any difference transparently rather than claiming full
#    design weighting of the final standardization step.
# 3. Locked mediator and rare-outcome note. iv_z_wt is the single prespecified
#    mediator. The Cox-based interventional effects are most stable when the
#    fixed-horizon event probability is not high; the Python audit uses a 60-month
#    risk-difference horizon for this reason.
