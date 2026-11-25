# ============================================================
# CR-DNN FINAL SEQUENTIAL ANALYSIS
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(readr)
})

# ============================================================
# PROJECT DIRECTORIES
# Run from the repository root.
# ============================================================

root <- normalizePath(".", mustWork = TRUE)

results_dir <- file.path(
  root,
  "results_sequential_master"
)

analysis_dir <- file.path(
  root,
  "final_analysis"
)

data_dir <- file.path(analysis_dir, "data")
table_dir <- file.path(analysis_dir, "tables")

results_dir <- file.path(root, "results_sequential_master")
analysis_dir <- file.path(root, "final_analysis")
data_dir <- file.path(analysis_dir, "data")
table_dir <- file.path(analysis_dir, "tables")

dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# 1. Locate all completed production batches
# ------------------------------------------------------------

files <- list.files(
  results_dir,
  pattern = "^neural__.*__rep_[0-9]+_[0-9]+\\.rds$",
  full.names = TRUE
)

files <- sort(files)

cat("Production RDS files:", length(files), "\n")

stopifnot(length(files) == 370L)

# ------------------------------------------------------------
# 2. Read all result objects
# ------------------------------------------------------------

objects <- map(files, readRDS)

required_components <- c(
  "raw",
  "selection",
  "tuning",
  "contrasts",
  "invariants",
  "dgp"
)

component_check <- map_lgl(
  objects,
  ~ all(required_components %in% names(.x))
)

if (!all(component_check)) {
  stop("At least one RDS object is missing a required component.")
}

# ------------------------------------------------------------
# 3. Bind components
# ------------------------------------------------------------

raw <- map_dfr(objects, "raw")
selection <- map_dfr(objects, "selection")
tuning <- map_dfr(objects, "tuning")
contrasts <- map_dfr(objects, "contrasts")
invariants <- map_dfr(objects, "invariants")
dgp_batches <- map_dfr(objects, "dgp")

cat("\nBound row counts:\n")
cat("raw        :", nrow(raw), "\n")
cat("selection  :", nrow(selection), "\n")
cat("tuning     :", nrow(tuning), "\n")
cat("contrasts  :", nrow(contrasts), "\n")
cat("invariants :", nrow(invariants), "\n")
cat("dgp batches:", nrow(dgp_batches), "\n")

# Strong production checks
stopifnot(nrow(raw) == 1480L)
stopifnot(nrow(selection) == 740L)
stopifnot(nrow(contrasts) == 740L)
stopifnot(nrow(invariants) == 740L)

# ------------------------------------------------------------
# 4. Core key and duplication checks
# ------------------------------------------------------------

rep_key <- c(
  "scenario",
  "n",
  "p",
  "p_over_n",
  "depth",
  "rep"
)

raw_duplicates <- raw |>
  count(
    across(all_of(c(rep_key, "method"))),
    name = "frequency"
  ) |>
  filter(frequency != 1L)

selection_duplicates <- selection |>
  count(
    across(all_of(rep_key)),
    name = "frequency"
  ) |>
  filter(frequency != 1L)

contrast_duplicates <- contrasts |>
  count(
    across(all_of(rep_key)),
    name = "frequency"
  ) |>
  filter(frequency != 1L)

cat("\nDuplicate-key checks:\n")
cat("raw duplicates      :", nrow(raw_duplicates), "\n")
cat("selection duplicates:", nrow(selection_duplicates), "\n")
cat("contrast duplicates :", nrow(contrast_duplicates), "\n")

stopifnot(nrow(raw_duplicates) == 0L)
stopifnot(nrow(selection_duplicates) == 0L)
stopifnot(nrow(contrast_duplicates) == 0L)

# Each replication should contain DNN and CR-DNN
method_check <- raw |>
  count(across(all_of(rep_key)), name = "n_methods")

stopifnot(all(method_check$n_methods == 2L))

# ------------------------------------------------------------
# 5. Unique DGP table: should be 54 neural cells
# ------------------------------------------------------------

dgp <- dgp_batches |>
  distinct()

cat("\nUnique DGP cells:", nrow(dgp), "\n")
stopifnot(nrow(dgp) == 54L)

write_csv(
  dgp,
  file.path(data_dir, "dgp_54_cells.csv")
)

# ------------------------------------------------------------
# 6. Reconstruct final B for each cell
# ------------------------------------------------------------

cell_B <- selection |>
  group_by(
    scenario,
    n,
    p,
    p_over_n,
    depth
  ) |>
  summarise(
    B_final = max(rep),
    n_replications = n(),
    .groups = "drop"
  ) |>
  mutate(
    # Final production log established that all nine B=20 cells
    # terminated at the maximum budget rather than precision.
    stop_reason = if_else(
      B_final == 20L,
      "stop_max",
      "stop_precision"
    )
  )

cat("\nFinal stopping distribution:\n")
print(
  cell_B |>
    count(B_final, stop_reason)
)

stopifnot(nrow(cell_B) == 54L)
stopifnot(sum(cell_B$n_replications) == 740L)
stopifnot(sum(cell_B$stop_reason == "stop_max") == 9L)
stopifnot(sum(cell_B$stop_reason == "stop_precision") == 45L)

write_csv(
  cell_B,
  file.path(data_dir, "final_cell_stopping.csv")
)

# ------------------------------------------------------------
# 7. Save complete analytical components
# ------------------------------------------------------------

write_csv(raw, file.path(data_dir, "replication_raw.csv"))
write_csv(selection, file.path(data_dir, "replication_selection.csv"))
write_csv(tuning, file.path(data_dir, "replication_tuning.csv"))
write_csv(contrasts, file.path(data_dir, "replication_contrasts.csv"))
write_csv(invariants, file.path(data_dir, "replication_invariants.csv"))

# ------------------------------------------------------------
# 8. Invariant checks
# ------------------------------------------------------------

invariant_summary <- invariants |>
  summarise(
    n_rows = n(),
    methods_min = min(methods),
    methods_max = max(methods),
    bayes_risks_min = min(bayes_risks),
    bayes_risks_max = max(bayes_risks),
    bayes_logistic_min = min(bayes_logistic_risks),
    bayes_logistic_max = max(bayes_logistic_risks),
    bayes_brier_min = min(bayes_briers),
    bayes_brier_max = max(bayes_briers)
  )

cat("\nInvariant summary:\n")
print(invariant_summary)

write_csv(
  invariant_summary,
  file.path(table_dir, "invariant_checks.csv")
)

# ------------------------------------------------------------
# 9. Alpha-selection summary
# ------------------------------------------------------------

alpha_selection <- selection |>
  mutate(
    alpha_class = case_when(
      cr_alpha_identity ~ "identity_alpha_1",
      cr_alpha_lower_boundary ~ "lower_boundary",
      cr_alpha_interior ~ "interior",
      TRUE ~ "other"
    )
  )

alpha_overall <- alpha_selection |>
  count(alpha_class, name = "n") |>
  mutate(
    proportion = n / sum(n),
    percent = 100 * proportion
  ) |>
  arrange(desc(n))

cat("\nOverall alpha-selection class:\n")
print(alpha_overall)

write_csv(
  alpha_overall,
  file.path(table_dir, "alpha_selection_overall.csv")
)

alpha_values <- selection |>
  count(cr_alpha, name = "n") |>
  arrange(cr_alpha) |>
  mutate(
    proportion = n / sum(n),
    percent = 100 * proportion
  )

cat("\nSelected alpha values:\n")
print(alpha_values)

write_csv(
  alpha_values,
  file.path(table_dir, "selected_alpha_values.csv")
)

# ------------------------------------------------------------
# 10. Alpha selection by simulation cell
# ------------------------------------------------------------

alpha_by_cell <- alpha_selection |>
  group_by(
    scenario,
    p_over_n,
    depth
  ) |>
  summarise(
    B = n(),
    mean_alpha = mean(cr_alpha, na.rm = TRUE),
    median_alpha = median(cr_alpha, na.rm = TRUE),
    prop_alpha_1 = mean(cr_alpha_identity, na.rm = TRUE),
    prop_alpha_interior = mean(cr_alpha_interior, na.rm = TRUE),
    prop_alpha_lower = mean(cr_alpha_lower_boundary, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  alpha_by_cell,
  file.path(table_dir, "alpha_selection_by_cell.csv")
)

# ------------------------------------------------------------
# 11. Method-level performance by cell
# ------------------------------------------------------------

performance_by_cell <- raw |>
  group_by(
    scenario,
    p_over_n,
    depth,
    method
  ) |>
  summarise(
    B = n(),
    mean_risk = mean(risk, na.rm = TRUE),
    sd_risk = sd(risk, na.rm = TRUE),
    
    mean_conditional_risk =
      mean(conditional_risk, na.rm = TRUE),
    
    mean_excess_class_risk =
      mean(excess_class_risk, na.rm = TRUE),
    
    sd_excess_class_risk =
      sd(excess_class_risk, na.rm = TRUE),
    
    mcse_excess_class_risk =
      sd_excess_class_risk / sqrt(B),
    
    mean_logistic_risk =
      mean(logistic_risk, na.rm = TRUE),
    
    mean_surrogate_excess_risk =
      mean(surrogate_excess_risk, na.rm = TRUE),
    
    mean_auc =
      mean(auc, na.rm = TRUE),
    
    sd_auc =
      sd(auc, na.rm = TRUE),
    
    mean_brier =
      mean(brier, na.rm = TRUE),
    
    mean_excess_brier =
      mean(excess_brier, na.rm = TRUE),
    
    mean_probability_sd =
      mean(probability_sd, na.rm = TRUE),
    
    mean_best_epoch =
      mean(best_epoch, na.rm = TRUE),
    
    mean_epochs_used =
      mean(epochs_used, na.rm = TRUE),
    
    mean_selected_alpha =
      mean(selected_alpha, na.rm = TRUE),
    
    mean_selected_lambda =
      mean(selected_lambda, na.rm = TRUE),
    
    mean_condition_number =
      if (all(is.na(condition_number))) {
        NA_real_
      } else {
        mean(condition_number, na.rm = TRUE)
      },
    
    .groups = "drop"
  )

write_csv(
  performance_by_cell,
  file.path(table_dir, "performance_by_cell_method.csv")
)

# ------------------------------------------------------------
# 12. Paired CR-DNN - DNN contrasts by cell
#
# Negative differences:
#   risk, conditional risk, excess risk, Brier, logistic risk
#   => favour CR-DNN.
#
# Positive AUC difference => favours CR-DNN.
# ------------------------------------------------------------

contrast_by_cell <- contrasts |>
  group_by(
    scenario,
    p_over_n,
    depth
  ) |>
  summarise(
    B = n(),
    
    mean_risk_diff =
      mean(risk_difference, na.rm = TRUE),
    sd_risk_diff =
      sd(risk_difference, na.rm = TRUE),
    mcse_risk_diff =
      sd_risk_diff / sqrt(B),
    
    mean_conditional_risk_diff =
      mean(conditional_risk_difference, na.rm = TRUE),
    
    mean_excess_risk_diff =
      mean(excess_risk_difference, na.rm = TRUE),
    sd_excess_risk_diff =
      sd(excess_risk_difference, na.rm = TRUE),
    mcse_excess_risk_diff =
      sd_excess_risk_diff / sqrt(B),
    
    mean_auc_diff =
      mean(auc_difference, na.rm = TRUE),
    sd_auc_diff =
      sd(auc_difference, na.rm = TRUE),
    mcse_auc_diff =
      sd_auc_diff / sqrt(B),
    
    mean_brier_diff =
      mean(brier_difference, na.rm = TRUE),
    sd_brier_diff =
      sd(brier_difference, na.rm = TRUE),
    mcse_brier_diff =
      sd_brier_diff / sqrt(B),
    
    mean_logistic_risk_diff =
      mean(logistic_risk_difference, na.rm = TRUE),
    
    prop_lower_excess_risk =
      mean(excess_risk_difference < 0, na.rm = TRUE),
    
    prop_higher_auc =
      mean(auc_difference > 0, na.rm = TRUE),
    
    prop_lower_brier =
      mean(brier_difference < 0, na.rm = TRUE),
    
    prop_exact_equal_excess =
      mean(excess_risk_difference == 0, na.rm = TRUE),
    
    .groups = "drop"
  )

write_csv(
  contrast_by_cell,
  file.path(table_dir, "paired_contrasts_by_cell.csv")
)

# ------------------------------------------------------------
# 13. Overall paired contrasts
# Descriptive only: design cells have unequal B by construction.
# ------------------------------------------------------------

contrast_overall_repweighted <- contrasts |>
  summarise(
    n_replications = n(),
    
    mean_excess_risk_diff =
      mean(excess_risk_difference, na.rm = TRUE),
    
    median_excess_risk_diff =
      median(excess_risk_difference, na.rm = TRUE),
    
    mean_auc_diff =
      mean(auc_difference, na.rm = TRUE),
    
    mean_brier_diff =
      mean(brier_difference, na.rm = TRUE),
    
    mean_logistic_risk_diff =
      mean(logistic_risk_difference, na.rm = TRUE),
    
    prop_lower_excess_risk =
      mean(excess_risk_difference < 0, na.rm = TRUE),
    
    prop_equal_excess_risk =
      mean(excess_risk_difference == 0, na.rm = TRUE),
    
    prop_higher_excess_risk =
      mean(excess_risk_difference > 0, na.rm = TRUE)
  )

write_csv(
  contrast_overall_repweighted,
  file.path(
    table_dir,
    "paired_contrasts_overall_repweighted.csv"
  )
)

# Equal-cell-weighted summary is preferable for an overall design summary
contrast_overall_cellweighted <- contrast_by_cell |>
  summarise(
    n_cells = n(),
    
    mean_cell_excess_risk_diff =
      mean(mean_excess_risk_diff, na.rm = TRUE),
    
    median_cell_excess_risk_diff =
      median(mean_excess_risk_diff, na.rm = TRUE),
    
    mean_cell_auc_diff =
      mean(mean_auc_diff, na.rm = TRUE),
    
    mean_cell_brier_diff =
      mean(mean_brier_diff, na.rm = TRUE),
    
    mean_cell_logistic_risk_diff =
      mean(mean_logistic_risk_diff, na.rm = TRUE),
    
    prop_cells_mean_excess_below_zero =
      mean(mean_excess_risk_diff < 0, na.rm = TRUE),
    
    prop_cells_mean_excess_equal_zero =
      mean(mean_excess_risk_diff == 0, na.rm = TRUE),
    
    prop_cells_mean_excess_above_zero =
      mean(mean_excess_risk_diff > 0, na.rm = TRUE)
  )

write_csv(
  contrast_overall_cellweighted,
  file.path(
    table_dir,
    "paired_contrasts_overall_cellweighted.csv"
  )
)

# ------------------------------------------------------------
# 14. Identity endpoint equivalence check
# ------------------------------------------------------------

identity_check <- raw |>
  filter(method == "CR-DNN", selected_alpha == 1) |>
  summarise(
    n_alpha_1 = n(),
    max_endpoint_probability_difference =
      max(endpoint_probability_difference, na.rm = TRUE),
    mean_endpoint_probability_difference =
      mean(endpoint_probability_difference, na.rm = TRUE)
  )

cat("\nIdentity-endpoint equivalence:\n")
print(identity_check)

write_csv(
  identity_check,
  file.path(table_dir, "identity_endpoint_equivalence.csv")
)

# ------------------------------------------------------------
# 15. Selected CR-DNN conditioning
# ------------------------------------------------------------

conditioning_selected <- raw |>
  filter(method == "CR-DNN") |>
  group_by(
    scenario,
    p_over_n,
    depth
  ) |>
  summarise(
    B = n(),
    mean_selected_alpha =
      mean(selected_alpha, na.rm = TRUE),
    median_selected_alpha =
      median(selected_alpha, na.rm = TRUE),
    mean_condition_number =
      mean(condition_number, na.rm = TRUE),
    median_condition_number =
      median(condition_number, na.rm = TRUE),
    max_condition_number =
      max(condition_number, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  conditioning_selected,
  file.path(table_dir, "selected_crdnn_conditioning.csv")
)

# ------------------------------------------------------------
# 16. Link selected alpha/conditioning to paired performance
# ------------------------------------------------------------

cr_selected <- raw |>
  filter(method == "CR-DNN") |>
  select(
    all_of(rep_key),
    selected_alpha,
    selected_lambda,
    condition_number,
    probability_sd,
    best_epoch,
    epochs_used
  )

joint_replication <- contrasts |>
  left_join(
    cr_selected,
    by = rep_key
  ) |>
  left_join(
    selection |>
      select(
        all_of(rep_key),
        cr_alpha_interior,
        cr_alpha_identity,
        cr_alpha_lower_boundary,
        cr_lambda_boundary,
        dnn_lambda_boundary,
        endpoint_probability_difference
      ),
    by = rep_key
  )

write_csv(
  joint_replication,
  file.path(data_dir, "joint_replication_analysis.csv")
)

# ------------------------------------------------------------
# 17. Tuning-grid summaries
# ------------------------------------------------------------

alpha_tuning_summary <- tuning |>
  group_by(alpha) |>
  summarise(
    n = n(),
    mean_cv_loss =
      mean(cv_validation_loss, na.rm = TRUE),
    mean_cv_auc =
      mean(cv_auc, na.rm = TRUE),
    mean_cv_brier =
      mean(cv_brier, na.rm = TRUE),
    mean_condition_number =
      mean(mean_condition_number, na.rm = TRUE),
    selected_cr_rate =
      mean(selected_cr, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(alpha)

write_csv(
  alpha_tuning_summary,
  file.path(table_dir, "tuning_summary_by_alpha.csv")
)

# ------------------------------------------------------------
# 18. Computational efficiency
# ------------------------------------------------------------

efficiency <- tibble(
  neural_cells = 54L,
  B_initial = 12L,
  B_max = 20L,
  fixed_B20_replications = 54L * 20L,
  sequential_replications = nrow(selection),
  replications_saved =
    fixed_B20_replications - sequential_replications,
  percent_saved =
    100 * replications_saved / fixed_B20_replications
)

cat("\nComputational efficiency:\n")
print(efficiency)

write_csv(
  efficiency,
  file.path(table_dir, "computational_efficiency_final.csv")
)

# ------------------------------------------------------------
# 19. Compact console report
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FINAL CR-DNN ANALYSIS DATASET COMPLETE\n")
cat("============================================\n")

cat("\nStopping distribution:\n")
print(
  cell_B |>
    count(B_final, stop_reason)
)

cat("\nAlpha selection:\n")
print(alpha_overall)

cat("\nSelected alpha values:\n")
print(alpha_values)

cat("\nEqual-cell-weighted paired summary:\n")
print(contrast_overall_cellweighted)

cat("\nReplication-weighted paired summary:\n")
print(contrast_overall_repweighted)

cat("\nIdentity endpoint check:\n")
print(identity_check)

cat("\nAll output written to:\n")
cat(analysis_dir, "\n")
