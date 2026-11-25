
# ============================================================
# CR-DNN FINAL PRODUCTION SIMULATION FUNCTIONS
# Reviewer-aligned production version
#
# Final benchmark set:
#   1. Shrinkage LDA
#   2. Ridge logistic regression
#   3. Elastic-net logistic regression
#   4. ANN / DNN
#   5. CR-DNN
#
# Key design features:
#   - scale-preserving CR-DNN transformation;
#   - alpha = 1 is exactly the ordinary dimension-normalised
#     ANN/DNN endpoint for a fixed lambda and random seed;
#   - all tuning is confined to the training sample;
#   - methods within a replication use the same generated
#     training/test data and common stratified CV folds;
#   - linear benchmarks are fitted once per scenario x p/n
#     cell because they do not depend on network depth;
#   - neural models are fitted for L in {1,3,5};
#   - L = 1 is labelled ANN in manuscript-facing output;
#   - production results are saved cell by cell.
# ============================================================

# ============================================================
# HPC PERFORMANCE CONFIGURATION
#
# Keep each SLURM task inside its allocated CPU budget. This
# prevents BLAS/OpenMP/TensorFlow oversubscription when many
# Monte Carlo replications run concurrently across the cluster.
# ============================================================

.crdnn_cpus <- suppressWarnings(
  as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1"))
)

if (
  length(.crdnn_cpus) != 1L ||
  is.na(.crdnn_cpus) ||
  .crdnn_cpus < 1L
) {
  .crdnn_cpus <- 1L
}

Sys.setenv(
  OMP_NUM_THREADS = as.character(.crdnn_cpus),
  OPENBLAS_NUM_THREADS = as.character(.crdnn_cpus),
  MKL_NUM_THREADS = as.character(.crdnn_cpus),
  NUMEXPR_NUM_THREADS = as.character(.crdnn_cpus),
  TF_NUM_INTRAOP_THREADS = as.character(.crdnn_cpus),
  TF_NUM_INTEROP_THREADS = "1",
  TF_CPP_MIN_LOG_LEVEL = "2"
)

options(keras.view_metrics = FALSE)

# Group many mini-batches inside each compiled backend call.  With
# our small dense networks this sharply reduces R/Python dispatch.
.crdnn_steps_per_execution <- suppressWarnings(
  as.integer(
    Sys.getenv(
      "CRDNN_STEPS_PER_EXECUTION",
      "32"
    )
  )
)

if (
  length(.crdnn_steps_per_execution) != 1L ||
  is.na(.crdnn_steps_per_execution) ||
  .crdnn_steps_per_execution < 1L
) {
  .crdnn_steps_per_execution <- 32L
}

# High-turnover simulations create a new small model for every
# tuning candidate.  Avoid repeated XLA compilation by default;
# users can set CRDNN_JIT_COMPILE=auto or true for a backend/GPU
# where a pilot benchmark shows that XLA is faster.
.crdnn_jit_text <- tolower(
  Sys.getenv(
    "CRDNN_JIT_COMPILE",
    "false"
  )
)

.crdnn_jit_compile <-
  if (
    identical(
      .crdnn_jit_text,
      "auto"
    )
  ) {
    "auto"
  } else {
    .crdnn_jit_text %in%
      c(
        "1",
        "true",
        "yes"
      )
  }

required_packages <- c(
  "keras3",
  "glmnet",
  "dplyr",
  "tibble",
  "tidyr"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0L) {
  stop(
    "Please install required packages: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

library(keras3)
library(glmnet)
library(dplyr)
library(tibble)
library(tidyr)

keras3::use_backend("tensorflow")


# ============================================================
# 0. GENERAL UTILITIES
# ============================================================


orthonormal_basis <- function(p, r) {

  A <- matrix(
    rnorm(p * r),
    nrow = p,
    ncol = r
  )

  Q <- qr.Q(
    qr(A)
  )

  Q[
    ,
    seq_len(r),
    drop = FALSE
  ]
}

safe_auc <- function(y, prob) {

  y <- as.integer(y)

  n1 <- sum(y == 1L)
  n0 <- sum(y == 0L)

  if (n1 == 0L || n0 == 0L) {
    return(NA_real_)
  }

  rr <- rank(
    prob,
    ties.method = "average"
  )

  (
    sum(rr[y == 1L]) -
      n1 * (n1 + 1) / 2
  ) / (n1 * n0)
}

set_model_seed <- function(seed) {

  set.seed(seed)

  try(
    keras3::set_random_seed(
      as.integer(seed)
    ),
    silent = TRUE
  )
}

make_stratified_foldid <- function(
    y,
    k = 5L,
    seed = 1L
) {

  y <- as.integer(y)

  set.seed(seed)

  foldid <- integer(
    length(y)
  )

  for (cls in sort(unique(y))) {

    idx <- which(y == cls)

    idx <- sample(idx)

    folds <- rep(
      seq_len(k),
      length.out = length(idx)
    )

    foldid[idx] <- folds
  }

  foldid
}

# ============================================================
# 1. FINAL SCENARIO SPECIFICATION
# ============================================================

get_scenario_spec <- function(scenario) {

  specs <- list(

    near_iso_nonlinear = list(
      covariance_regime = "near_iso",
      response_regime = "nonlinear",
      geometry_level = "near_isotropic",
      signal_multiplier = 0.400,
      target_bayes_risk = 0.17,
      scenario_role = "principal"
    ),

    moderate_aniso_nonlinear = list(
      covariance_regime = "moderate_aniso",
      response_regime = "nonlinear",
      geometry_level = "moderate_anisotropy",
      signal_multiplier = 0.400,
      target_bayes_risk = 0.17,
      scenario_role = "geometry_contrast"
    ),

    strong_aniso_nonlinear = list(
      covariance_regime = "strong_aniso",
      response_regime = "nonlinear",
      geometry_level = "strong_anisotropy",
      signal_multiplier = 0.400,
      target_bayes_risk = 0.17,
      scenario_role = "geometry_stress"
    ),

    near_iso_linear = list(
      covariance_regime = "near_iso",
      response_regime = "linear",
      geometry_level = "near_isotropic",
      signal_multiplier = 0.525,
      target_bayes_risk = 0.17,
      scenario_role = "response_contrast"
    ),

    moderate_aniso_linear = list(
      covariance_regime = "moderate_aniso",
      response_regime = "linear",
      geometry_level = "moderate_anisotropy",
      signal_multiplier = 0.525,
      target_bayes_risk = 0.17,
      scenario_role = "response_geometry_contrast"
    ),

    strong_aniso_linear = list(
      covariance_regime = "strong_aniso",
      response_regime = "linear",
      geometry_level = "strong_anisotropy",
      signal_multiplier = 0.525,
      target_bayes_risk = 0.17,
      scenario_role = "response_geometry_stress"
    ),

    near_iso_nonlinear_weak = list(
      covariance_regime = "near_iso",
      response_regime = "nonlinear",
      geometry_level = "near_isotropic",
      signal_multiplier = 0.200,
      target_bayes_risk = 0.27,
      scenario_role = "weak_signal_robustness"
    )
  )

  if (
    length(scenario) != 1L ||
    !is.character(scenario) ||
    !scenario %in% names(specs)
  ) {
    stop(
      "Unknown simulation scenario: ",
      scenario,
      call. = FALSE
    )
  }

  specs[[scenario]]
}


interaction_map <- function(
    H,
    gamma = 1
) {

  H <- as.matrix(
    H
  )

  n <- nrow(
    H
  )

  r <- ncol(
    H
  )


  if (
    r < 2L
  ) {

    stop(
      "interaction_map requires at least two latent coordinates.",
      call. = FALSE
    )
  }


  Phi <- matrix(
    0,
    nrow = n,
    ncol = r
  )


  for (
    j in seq_len(r)
  ) {

    j_next <-
      if (
        j == r
      ) {
        1L
      } else {
        j + 1L
      }


    Phi[, j] <-
      tanh(
        gamma *
          H[, j] *
          H[, j_next]
      )
  }


  Phi
}

make_dgp <- function(
    p,
    r = 8L,
    q_true = 3L,   # retained temporarily for backward compatibility
    sigma2 = 1.5,
    base_signal = 2.0,
    scenario = c(
      "near_iso_nonlinear",
      "moderate_aniso_nonlinear",
      "strong_aniso_nonlinear",
      "near_iso_linear",
      "moderate_aniso_linear",
      "strong_aniso_linear",
      "near_iso_nonlinear_weak"
    ),
    signal_multiplier = NULL,
    seed = 2026L
) {

  # ----------------------------------------------------------
  # 2.0 Input validation
  # ----------------------------------------------------------

  scenario <- match.arg(scenario)

  if (
    length(p) != 1L ||
    !is.finite(p) ||
    p < 2L
  ) {
    stop(
      "p must be a finite integer greater than or equal to 2.",
      call. = FALSE
    )
  }

  p <- as.integer(p)


  if (
    length(r) != 1L ||
    r < 2L ||
    r > p
  ) {
    stop(
      "r must satisfy 2 <= r <= p.",
      call. = FALSE
    )
  }

  r <- as.integer(r)


  if (
    length(sigma2) != 1L ||
    !is.finite(sigma2) ||
    sigma2 <= 0
  ) {
    stop(
      "sigma2 must be strictly positive.",
      call. = FALSE
    )
  }


  if (
    length(base_signal) != 1L ||
    !is.finite(base_signal) ||
    base_signal <= 0
  ) {
    stop(
      "base_signal must be strictly positive.",
      call. = FALSE
    )
  }


  spec <-
    get_scenario_spec(
      scenario
    )


  # ----------------------------------------------------------
  # Allow a previously calibrated multiplier to replace the
  # default value stored in the scenario specification.
  # ----------------------------------------------------------

  if (
    is.null(signal_multiplier)
  ) {

    signal_multiplier <-
      spec$signal_multiplier

  } else {

    if (
      length(signal_multiplier) != 1L ||
      !is.finite(signal_multiplier) ||
      signal_multiplier <= 0
    ) {
      stop(
        "signal_multiplier must be strictly positive.",
        call. = FALSE
      )
    }
  }


  set.seed(seed)


  # ==========================================================
  # 2.1 Predictive subspace
  #
  # The predictive subspace is generated independently of the
  # covariance perturbation subspace. Hence any advantage from
  # covariance regularisation is not created by mechanically
  # aligning predictive and covariance directions.
  # ==========================================================

  U <-
    orthonormal_basis(
      p = p,
      r = r
    )


  # ==========================================================
  # 2.2 Covariance perturbation subspace
  # ==========================================================

  rank_delta <-
    min(
      10L,
      p
    )


  Q <-
    orthonormal_basis(
      p = p,
      r = rank_delta
    )


  # ==========================================================
  # 2.3 Controlled population covariance geometry
  #
  #     Sigma_p = sigma^2 I_p + Delta_p
  #
  # with
  #
  #     Delta_p = Q diag(d_1,...,d_k) Q^T.
  #
  # The three regimes are distinguished through the relative
  # operator perturbation
  #
  #     ||Delta_p||_op / sigma^2.
  #
  # For the principal near-isotropic regime this perturbation
  # decreases with p, whereas the anisotropic departures remain
  # non-vanishing.
  # ==========================================================

  if (
    spec$covariance_regime == "near_iso"
  ) {

    # --------------------------------------------------------
    # Theory-aligned near-isotropy:
    #
    #   ||Delta_p||_op / sigma^2
    #       = c_delta * p^{-1/4} -> 0.
    #
    # c_delta = 0.75 reproduces approximately the previous
    # perturbation magnitude near p = 1000:
    #
    #   d_max approximately 0.20 when sigma2 = 1.5.
    # --------------------------------------------------------

    c_delta <-
      0.75


    relative_delta <-
      c_delta *
      p^(-1 / 4)


    d_max <-
      sigma2 *
      relative_delta


    d_min <-
      0.50 *
      d_max


  } else if (
    spec$covariance_regime == "moderate_aniso"
  ) {

    # --------------------------------------------------------
    # Moderate non-vanishing anisotropy:
    #
    #   lambda_max(Sigma) / lambda_min(Sigma)
    #       approximately 2.
    # --------------------------------------------------------

    relative_delta <-
      1.0


    d_max <-
      sigma2 *
      relative_delta


    d_min <-
      0.25 *
      d_max


  } else if (
    spec$covariance_regime == "strong_aniso"
  ) {

    # --------------------------------------------------------
    # Strong non-vanishing anisotropy:
    #
    #   lambda_max(Sigma) / lambda_min(Sigma)
    #       approximately 5.
    # --------------------------------------------------------

    relative_delta <-
      4.0


    d_max <-
      sigma2 *
      relative_delta


    d_min <-
      0.25 *
      d_max


  } else {

    stop(
      "Unknown covariance regime: ",
      spec$covariance_regime,
      call. = FALSE
    )
  }


  d_vals <-
    seq(
      from = d_max,
      to = d_min,
      length.out = rank_delta
    )


  # ----------------------------------------------------------
  # FAST PRODUCTION PATH
  #
  # Neither Delta nor Sigma is used by sampling or by any
  # downstream production calculation. Sampling uses the exact
  # low-rank representation through Q and d_vals. Avoiding these
  # dense p x p matrices removes unnecessary memory and work.
  # ----------------------------------------------------------

  Delta <- NULL
  Sigma <- NULL


  # ==========================================================
  # 2.4 Response-structure parameters
  #
  # These parameters are held fixed across covariance regimes.
  #
  # For the nonlinear response:
  #
  #   eta(X)
  #     =
  #     linear_weight     * eta_linear(X)
  #     +
  #     nonlinear_weight  * eta_interaction(X).
  #
  # The calibrated principal nonlinear structure uses
  #
  #   linear_weight     = 1,
  #   nonlinear_weight  = 2,
  #   interaction_gamma = 1.
  # ==========================================================

  interaction_gamma <-
    1.0


  linear_weight <-
    1.0


  nonlinear_weight <-
    2.0


  # ==========================================================
  # 2.5 Signal strength
  #
  # Signal calibration is separated from covariance geometry.
  # Each final scenario can therefore be calibrated to a common
  # target Bayes risk before the definitive Monte Carlo study.
  # ==========================================================

  effective_signal <-
    base_signal *
    signal_multiplier


  beta <-
    seq(
      from = effective_signal,
      to = effective_signal / 2,
      length.out = r
    ) *
    sample(
      c(-1, 1),
      size = r,
      replace = TRUE
    )


  # ==========================================================
  # 2.6 Population geometry diagnostics
  # ==========================================================

  delta_operator <-
    max(
      abs(
        d_vals
      )
    )


  delta_relative <-
    delta_operator /
    sigma2


  # Since Delta is positive semidefinite and rank deficient,
  # the smallest population eigenvalue remains sigma2.
  kappa_population <-
    (
      sigma2 +
        max(d_vals)
    ) /
    sigma2


  lambda_min_population <-
    sigma2


  lambda_max_population <-
    sigma2 +
    max(d_vals)


  # ----------------------------------------------------------
  # Average marginal variance.
  #
  # This is retained as a diagnostic because the low-rank
  # perturbation changes total variance slightly across regimes.
  # Bayes-risk calibration will subsequently control intrinsic
  # prediction difficulty.
  # ----------------------------------------------------------

  mean_population_variance <-
    sigma2 +
    sum(d_vals) /
    p


  # ==========================================================
  # 2.7 Return fixed structural DGP
  # ==========================================================

  list(

    p = p,

    r = r,

    # q_true is no longer part of the principal response
    # construction, but is retained temporarily so that older
    # downstream code does not fail.
    q_true = q_true,

    sigma2 = sigma2,

    scenario = scenario,

    scenario_role =
      spec$scenario_role,

    covariance_regime =
      spec$covariance_regime,

    geometry_level =
      spec$geometry_level,

    response_regime =
      spec$response_regime,

    target_bayes_risk =
      spec$target_bayes_risk,

    signal_multiplier =
      signal_multiplier,

    effective_signal =
      effective_signal,

    U = U,

    Q = Q,

    rank_delta =
      rank_delta,

    d_vals =
      d_vals,

    Delta =
      Delta,

    Sigma =
      Sigma,

    interaction_gamma =
      interaction_gamma,

    linear_weight =
      linear_weight,

    nonlinear_weight =
      nonlinear_weight,

    beta =
      beta,

    delta_operator =
      delta_operator,

    delta_relative =
      delta_relative,

    kappa_population =
      kappa_population,

    lambda_min_population =
      lambda_min_population,

    lambda_max_population =
      lambda_max_population,

    mean_population_variance =
      mean_population_variance
  )
}

sample_from_dgp <- function(
    n,
    dgp,
    seed = NULL
) {

  # ----------------------------------------------------------
  # 3.0 Input validation
  # ----------------------------------------------------------

  if (
    length(n) != 1L ||
    !is.finite(n) ||
    n < 1L
  ) {
    stop(
      "n must be a positive integer.",
      call. = FALSE
    )
  }

  n <-
    as.integer(n)


  if (
    !is.null(seed)
  ) {

    set.seed(
      seed
    )
  }


  p <-
    dgp$p


  if (
    !dgp$response_regime %in%
    c(
      "linear",
      "nonlinear"
    )
  ) {
    stop(
      "Unknown response regime: ",
      dgp$response_regime,
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        dgp$d_vals
      )
    ) ||
    any(
      dgp$d_vals < 0
    )
  ) {
    stop(
      "All covariance perturbation eigenvalues must be finite and non-negative.",
      call. = FALSE
    )
  }


  # ==========================================================
  # 3.1 Predictor generation
  #
  # Population covariance:
  #
  #   Sigma
  #     =
  #     sigma^2 I_p
  #     +
  #     Q diag(d_1,...,d_k) Q^T.
  #
  # Rather than forming a new Cholesky factor of the full
  # p x p covariance matrix for every simulated sample, use
  # the exact stochastic representation
  #
  #   X
  #     =
  #     sigma Z
  #     +
  #     G diag(sqrt(d_1),...,sqrt(d_k)) Q^T,
  #
  # where
  #
  #   Z ~ N(0,I_p),
  #   G ~ N(0,I_k),
  #
  # independently.
  #
  # Therefore
  #
  #   Cov(X)
  #     =
  #     sigma^2 I_p
  #     +
  #     Q diag(d) Q^T
  #     =
  #     Sigma.
  #
  # This representation is exact and computationally much
  # cheaper when rank_delta << p.
  # ==========================================================

  Z <-
    matrix(
      rnorm(
        n * p
      ),
      nrow = n,
      ncol = p
    )


  rank_delta <-
    length(
      dgp$d_vals
    )


  G <-
    matrix(
      rnorm(
        n * rank_delta
      ),
      nrow = n,
      ncol = rank_delta
    )


  G_scaled <-
    sweep(
      G,
      MARGIN = 2L,
      STATS = sqrt(
        dgp$d_vals
      ),
      FUN = "*"
    )


  X <-
    sqrt(
      dgp$sigma2
    ) *
    Z +

    G_scaled %*%
    t(
      dgp$Q
    )


  # ==========================================================
  # 3.2 Predictive latent representation
  #
  # For row-wise observations:
  #
  #   H0 = X U.
  #
  # U is generated independently of the covariance
  # perturbation subspace Q.
  # ==========================================================

  H0 <-
    X %*%
    dgp$U


  # ==========================================================
  # 3.3 Linear score component
  # ==========================================================

  eta_linear <-
    drop(
      H0 %*%
      dgp$beta
    )


  # ==========================================================
  # 3.4 Response mechanism
  #
  # LINEAR:
  #
  #   eta(X)
  #     =
  #     a eta_linear(X).
  #
  # NONLINEAR:
  #
  #   eta(X)
  #     =
  #     a eta_linear(X)
  #     +
  #     b eta_nonlinear(X),
  #
  # where
  #
  #   eta_nonlinear(X)
  #     =
  #     Phi(H0)^T beta.
  #
  # The same latent predictive representation and coefficient
  # vector beta are retained across response regimes.
  # ==========================================================

  if (
    dgp$response_regime == "nonlinear"
  ) {

    H_int <-
      interaction_map(
        H =
          H0,
        gamma =
          dgp$interaction_gamma
      )


    eta_nonlinear <-
      drop(
        H_int %*%
        dgp$beta
      )


    eta_linear_weighted <-
      dgp$linear_weight *
      eta_linear


    eta_nonlinear_weighted <-
      dgp$nonlinear_weight *
      eta_nonlinear


    eta <-
      eta_linear_weighted +
      eta_nonlinear_weighted


  } else if (
    dgp$response_regime == "linear"
  ) {

    # --------------------------------------------------------
    # Matched purely linear response regime
    # --------------------------------------------------------

    H_int <-
      NULL


    eta_nonlinear <-
      rep(
        0,
        n
      )


    eta_linear_weighted <-
      dgp$linear_weight *
      eta_linear


    eta_nonlinear_weighted <-
      rep(
        0,
        n
      )


    eta <-
      eta_linear_weighted
  }


  # ==========================================================
  # 3.5 True conditional probability
  #
  #   pi(X)
  #     =
  #     P(Y=1 | X)
  #     =
  #     logistic{eta(X)}.
  # ==========================================================

  true_prob <-
    plogis(
      eta
    )


  if (
    any(
      !is.finite(
        true_prob
      )
    )
  ) {
    stop(
      "Non-finite conditional probabilities were generated.",
      call. = FALSE
    )
  }


  # ==========================================================
  # 3.6 Binary outcome
  #
  # Generate the Bernoulli response through an explicit
  # uniform variate.
  #
  # With common seeds, this provides common random numbers
  # across matched simulation scenarios and therefore improves
  # the precision of paired Monte Carlo contrasts.
  # ==========================================================

  outcome_uniform <-
    runif(
      n
    )


  y <-
    as.integer(
      outcome_uniform <
      true_prob
    )


  # ==========================================================
  # 3.7 Bayes classifier
  # ==========================================================

  bayes_class <-
    as.integer(
      true_prob >=
      0.5
    )


  # ==========================================================
  # 3.8 Return
  #
  # Keep the unweighted and weighted score components
  # explicitly so that the structural DGP remains auditable.
  # ==========================================================

  list(

    X =
      X,

    y =
      y,

    true_prob =
      true_prob,

    bayes_class =
      bayes_class,

    eta =
      eta,

    eta_linear =
      eta_linear,

    eta_nonlinear =
      eta_nonlinear,

    eta_linear_weighted =
      eta_linear_weighted,

    eta_nonlinear_weighted =
      eta_nonlinear_weighted,

    latent =
      H0,

    latent_interaction =
      H_int,

    outcome_uniform =
      outcome_uniform
  )
}

centre_by_train <- function(
    X_train,
    X_test,
    dimension_normalise = FALSE
) {

  # ----------------------------------------------------------
  # Convert to numeric matrices
  # ----------------------------------------------------------

  X_train <-
    as.matrix(
      X_train
    )


  X_test <-
    as.matrix(
      X_test
    )


  # ----------------------------------------------------------
  # Validate dimensions
  # ----------------------------------------------------------

  if (
    nrow(X_train) < 1L ||
    ncol(X_train) < 1L
  ) {

    stop(
      "X_train must contain at least one observation and one predictor.",
      call. = FALSE
    )
  }


  if (
    nrow(X_test) < 1L
  ) {

    stop(
      "X_test must contain at least one observation.",
      call. = FALSE
    )
  }


  if (
    ncol(X_test) !=
    ncol(X_train)
  ) {

    stop(
      "X_train and X_test must contain the same number of predictors.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        X_train
      )
    ) ||
    any(
      !is.finite(
        X_test
      )
    )
  ) {

    stop(
      "X_train and X_test must contain only finite values.",
      call. = FALSE
    )
  }


  if (
    length(
      dimension_normalise
    ) != 1L ||
    !is.logical(
      dimension_normalise
    ) ||
    is.na(
      dimension_normalise
    )
  ) {

    stop(
      "dimension_normalise must be TRUE or FALSE.",
      call. = FALSE
    )
  }


  p <-
    ncol(
      X_train
    )


  # ==========================================================
  # 5.1 Training-sample mean
  #
  # No information from the test sample enters preprocessing.
  # ==========================================================

  xbar <-
    colMeans(
      X_train
    )


  # ==========================================================
  # 5.2 Centre training and test predictors
  #
  # The SAME training-sample mean is used for both.
  # ==========================================================

  Xc_train <-
    sweep(
      X_train,
      MARGIN = 2L,
      STATS = xbar,
      FUN = "-"
    )


  Xc_test <-
    sweep(
      X_test,
      MARGIN = 2L,
      STATS = xbar,
      FUN = "-"
    )


  # ==========================================================
  # 5.3 Optional dimension normalisation
  #
  # For the scale-matched DNN comparator:
  #
  #       X_DNN = p^{-1/2}(X - Xbar_train).
  #
  # This is deliberately distinct from CR-DNN:
  #
  #       X_CR
  #         =
  #         p^{-1/2}
  #         Sigma_tilde_alpha^{-1/2}
  #         (X - Xbar_train).
  #
  # Hence the comparison isolates the directional covariance
  # transformation rather than a trivial difference in overall
  # input magnitude.
  # ==========================================================

  if (
    dimension_normalise
  ) {

    dimension_scale <-
      1 /
      sqrt(
        p
      )


    Xc_train <-
      dimension_scale *
      Xc_train


    Xc_test <-
      dimension_scale *
      Xc_test

  } else {

    dimension_scale <-
      1.0
  }


  # ==========================================================
  # 5.4 Return
  # ==========================================================

  list(

    X_train =
      Xc_train,

    X_test =
      Xc_test,

    xbar =
      xbar,

    p =
      p,

    dimension_normalise =
      dimension_normalise,

    dimension_scale =
      dimension_scale
  )
}

prepare_shrinkage <- function(
    X_train
) {

  X_train <- as.matrix(
    X_train
  )


  # ----------------------------------------------------------
  # Basic validation
  # ----------------------------------------------------------

  if (
    nrow(X_train) < 2L
  ) {

    stop(
      "X_train must contain at least two observations.",
      call. = FALSE
    )
  }


  if (
    ncol(X_train) < 1L
  ) {

    stop(
      "X_train must contain at least one predictor.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        X_train
      )
    )
  ) {

    stop(
      "X_train contains non-finite values.",
      call. = FALSE
    )
  }


  # ----------------------------------------------------------
  # Training-sample centring
  # ----------------------------------------------------------

  xbar <- colMeans(
    X_train
  )


  Xc <- sweep(
    X_train,
    2,
    xbar,
    FUN = "-"
  )


  n <- nrow(
    Xc
  )


  p <- ncol(
    Xc
  )


  # ----------------------------------------------------------
  # Shrinkage target
  #
  # Sigma_hat = n^{-1} Xc^T Xc
  #
  # tau_n
  #   = tr(Sigma_hat) / p
  #   = sum(Xc^2) / (n p)
  # ----------------------------------------------------------

  tau_cov <-
    sum(
      Xc^2
    ) /
    (n * p)


  if (
    !is.finite(tau_cov) ||
    tau_cov <= 0
  ) {

    stop(
      "Invalid shrinkage target: tau_cov must be positive.",
      call. = FALSE
    )
  }


  # ----------------------------------------------------------
  # Thin SVD
  #
  # If
  #
  #     Xc = U D V^T,
  #
  # then the non-zero eigenvalues of Sigma_hat are
  #
  #     d_j^2 / n.
  #
  # This avoids forming the full p x p covariance matrix.
  # ----------------------------------------------------------

  ss <- svd(
    Xc,
    nu = 0,
    nv = min(
      n,
      p
    )
  )


  if (
    length(
      ss$d
    ) == 0L
  ) {

    return(
      list(

        xbar =
          xbar,

        V =
          matrix(
            numeric(0),
            nrow = p,
            ncol = 0L
          ),

        eigvals =
          numeric(0),

        tau_cov =
          tau_cov,

        n =
          n,

        p =
          p
      )
    )
  }


  # ----------------------------------------------------------
  # Numerical rank threshold
  # ----------------------------------------------------------

  tol <-
    max(
      dim(
        Xc
      )
    ) *
    max(
      ss$d
    ) *
    .Machine$double.eps


  keep <-
    ss$d >
    tol


  eigvals <-
    (
      ss$d[keep]^2
    ) /
    n


  V <- ss$v[
    ,
    keep,
    drop = FALSE
  ]


  # ----------------------------------------------------------
  # Return
  # ----------------------------------------------------------

  list(

    xbar =
      xbar,

    V =
      V,

    eigvals =
      eigvals,

    tau_cov =
      tau_cov,

    n =
      n,

    p =
      p
  )
}

# ============================================================
# 7.2 EXACT SHRINKAGE CONDITION NUMBER
#
# For p < n the empirical covariance can be full rank, so the
# smallest regularised eigenvalue need not be alpha * tau.
# This helper reports the exact condition number in either the
# full-rank or rank-deficient case.
# ============================================================

shrinkage_condition_number <- function(
    eigvals,
    p,
    alpha,
    tau_cov
) {

  eigvals <- as.numeric(eigvals)
  p <- as.integer(p)

  c0 <- alpha * tau_cov

  if (!is.finite(c0) || c0 <= 0) {
    stop(
      "alpha * tau_cov must be positive.",
      call. = FALSE
    )
  }

  if (length(eigvals) == 0L) {
    return(1)
  }

  shrunk <- c0 + (1 - alpha) * eigvals
  max_eigen <- max(shrunk)

  min_eigen <-
    if (length(eigvals) < p) {
      c0
    } else {
      min(shrunk)
    }

  as.numeric(max_eigen / min_eigen)
}


# ============================================================
# 7.3 SCALE-PRESERVING SHRINKAGE TRANSFORMATION
#
# Sigma_tilde(alpha)
#   = (1-alpha) Sigma_hat + alpha tau_n I_p.
#
# Raw whitening changes both geometry and global scale.  We
# therefore multiply by a scalar c_alpha so that the empirical
# average input energy equals that of (X-Xbar)/sqrt(p).
#
# Consequently:
#
#   alpha = 1
#       => Z = (X-Xbar)/sqrt(p)
#
# exactly, so alpha = 1 is the ordinary ANN/DNN endpoint.
# ============================================================

apply_shrinkage <- function(
    prep,
    X,
    alpha
) {

  X <- as.matrix(X)
  alpha <- as.numeric(alpha)

  if (
    length(alpha) != 1L ||
    !is.finite(alpha) ||
    alpha <= 0 ||
    alpha > 1
  ) {
    stop(
      "alpha must satisfy 0 < alpha <= 1.",
      call. = FALSE
    )
  }

  if (ncol(X) != prep$p) {
    stop(
      "X must have the same number of predictors as prep.",
      call. = FALSE
    )
  }

  if (any(!is.finite(X))) {
    stop(
      "X contains non-finite values.",
      call. = FALSE
    )
  }

  Xc <- sweep(
    X,
    2,
    prep$xbar,
    FUN = "-"
  )

  # Exact identity endpoint.  At alpha = 1, the scale-preserving
  # CR-DNN transform is exactly the ordinary dimension-normalised
  # centred representation.  Return it directly rather than recovering
  # the same quantity through eigenspace arithmetic, which can introduce
  # unnecessary floating-point differences.
  if (isTRUE(all.equal(alpha, 1))) {
    return(
      list(
        Z = Xc / sqrt(prep$p),
        scale_match = sqrt(prep$tau_cov),
        transformed_variance = prep$tau_cov,
        condition_number = 1
      )
    )
  }

  c0 <- alpha * prep$tau_cov

  if (!is.finite(c0) || c0 <= 0) {
    stop(
      "alpha * tau_cov must be positive.",
      call. = FALSE
    )
  }

  base_scale <- 1 / sqrt(c0)

  if (length(prep$eigvals) == 0L) {

    Z_raw <- base_scale * Xc

  } else {

    projected <- Xc %*% prep$V

    shrunk_eigenvalues <-
      c0 +
      (1 - alpha) *
      prep$eigvals

    adjustment <-
      1 / sqrt(shrunk_eigenvalues) -
      base_scale

    Z_raw <-
      base_scale * Xc +
      sweep(
        projected,
        2,
        adjustment,
        FUN = "*"
      ) %*%
      t(prep$V)
  }

  # Average empirical variance after raw whitening:
  # p^{-1} tr(Sigma_tilde^{-1} Sigma_hat).
  if (length(prep$eigvals) == 0L) {

    whitened_variance <- 0

  } else {

    shrunk_eigenvalues <-
      c0 +
      (1 - alpha) *
      prep$eigvals

    whitened_variance <-
      sum(
        prep$eigvals /
        shrunk_eigenvalues
      ) /
      prep$p
  }

  if (
    !is.finite(whitened_variance) ||
    whitened_variance <= 0
  ) {
    stop(
      "Invalid transformed empirical variance.",
      call. = FALSE
    )
  }

  scale_match <-
    sqrt(
      prep$tau_cov /
      whitened_variance
    )

  Z <-
    scale_match *
    Z_raw /
    sqrt(prep$p)

  condition_number <-
    shrinkage_condition_number(
      eigvals = prep$eigvals,
      p = prep$p,
      alpha = alpha,
      tau_cov = prep$tau_cov
    )

  list(
    Z = Z,
    scale_match = scale_match,
    transformed_variance =
      whitened_variance *
      scale_match^2,
    condition_number = condition_number
  )
}


build_nn <- function(
    input_dim,
    depth,
    width,
    lambda_nn,
    penalty_tau = 1,
    learning_rate = 1e-3
) {

  # ----------------------------------------------------------
  # Validate architecture and optimisation parameters
  # ----------------------------------------------------------

  input_dim <-
    as.integer(
      input_dim
    )


  depth <-
    as.integer(
      depth
    )


  width <-
    as.integer(
      width
    )


  if (
    length(input_dim) != 1L ||
    !is.finite(input_dim) ||
    input_dim < 1L
  ) {

    stop(
      "input_dim must be a positive integer.",
      call. = FALSE
    )
  }


  if (
    length(depth) != 1L ||
    !is.finite(depth) ||
    depth < 1L
  ) {

    stop(
      "depth must be a positive integer.",
      call. = FALSE
    )
  }


  if (
    length(width) != 1L ||
    !is.finite(width) ||
    width < 1L
  ) {

    stop(
      "width must be a positive integer.",
      call. = FALSE
    )
  }


  if (
    length(lambda_nn) != 1L ||
    !is.finite(lambda_nn) ||
    lambda_nn < 0
  ) {

    stop(
      "lambda_nn must be a finite non-negative scalar.",
      call. = FALSE
    )
  }


  if (
    length(penalty_tau) != 1L ||
    !is.finite(penalty_tau) ||
    penalty_tau < 0
  ) {

    stop(
      "penalty_tau must be a finite non-negative scalar.",
      call. = FALSE
    )
  }


  if (
    length(learning_rate) != 1L ||
    !is.finite(learning_rate) ||
    learning_rate <= 0
  ) {

    stop(
      "learning_rate must be a positive finite scalar.",
      call. = FALSE
    )
  }


  # ----------------------------------------------------------
  # Input layer
  # ----------------------------------------------------------

  input <- keras3::layer_input(

    shape =
      c(
        input_dim
      ),

    dtype =
      "float32"
  )


  x <-
    input


  # ----------------------------------------------------------
  # Hidden layers
  #
  # Each hidden layer uses:
  #
  #   - fixed width;
  #   - ReLU activation;
  #   - common L1/L2 kernel regularisation.
  #
  # The regularisation coefficients are
  #
  #   lambda_L1 = penalty_tau * lambda_nn,
  #   lambda_L2 = lambda_nn.
  # ----------------------------------------------------------

  for (
    r_layer in seq_len(
      depth
    )
  ) {

    x <- keras3::layer_dense(

      object =
        x,

      units =
        width,

      activation =
        "relu",

      kernel_regularizer =
        keras3::regularizer_l1_l2(

          l1 =
            as.numeric(
              penalty_tau *
                lambda_nn
            ),

          l2 =
            as.numeric(
              lambda_nn
            )
        )
    )
  }


  # ----------------------------------------------------------
  # Binary-output layer
  #
  # No additional kernel penalty is imposed on the output
  # layer in this preliminary diagnostic.
  # ----------------------------------------------------------

  output <- keras3::layer_dense(

    object =
      x,

    units =
      1L,

    activation =
      "sigmoid"
  )


  # ----------------------------------------------------------
  # Model
  # ----------------------------------------------------------

  model <- keras3::keras_model(

    inputs =
      input,

    outputs =
      output
  )


  # ----------------------------------------------------------
  # Optimisation
  # ----------------------------------------------------------

  model$compile(

    optimizer =
      keras3::optimizer_adam(

        learning_rate =
          as.numeric(
            learning_rate
          )
      ),

    loss =
      "binary_crossentropy",

    # Only BCE is required for early stopping and diagnostics.
    # Avoid computing an unused accuracy metric on every batch.
    metrics =
      list(
        "binary_crossentropy"
      ),

    run_eagerly = FALSE,

    steps_per_execution =
      as.integer(
        .crdnn_steps_per_execution
      ),

    jit_compile =
      .crdnn_jit_compile
  )


  model
}


# ============================================================
# FIT AND PREDICT NEURAL NETWORK
#
# Used by both:
#   1. ordinary DNN/ANN endpoint fits;
#   2. CR-DNN tuning and final fits.
#
# Early stopping is based on validation binary cross-entropy,
# not penalised training loss.
# ============================================================

fit_predict_nn <- function(
    X_train,
    y_train,
    X_test,
    depth,
    width,
    lambda_nn,
    penalty_tau = 1,
    learning_rate = 1e-3,
    epochs = 100L,
    batch_size = 32L,
    validation_split = 0.10,
    patience = 10L,
    seed = 1L,
    verbose = FALSE
) {

  # ----------------------------------------------------------
  # 1. Input preparation
  # ----------------------------------------------------------

  X_train <-
    as.matrix(X_train)

  X_test <-
    as.matrix(X_test)

  storage.mode(X_train) <-
    "double"

  storage.mode(X_test) <-
    "double"


  y_train <-
    as.numeric(y_train)

  if (
    nrow(X_train) !=
    length(y_train)
  ) {

    stop(
      "X_train and y_train have incompatible dimensions.",
      call. = FALSE
    )
  }


  if (
    any(!is.finite(X_train)) ||
    any(!is.finite(X_test)) ||
    any(!is.finite(y_train))
  ) {

    stop(
      "Non-finite values supplied to fit_predict_nn().",
      call. = FALSE
    )
  }


  if (
    !all(
      y_train %in%
        c(0, 1)
    )
  ) {

    stop(
      "y_train must be binary and coded 0/1.",
      call. = FALSE
    )
  }


  # Keras 3 binary response:
  # n x 1 numeric array

  y_array <-
    array(
      y_train,
      dim =
        c(
          length(y_train),
          1L
        )
    )

  storage.mode(y_array) <-
    "double"


  # ----------------------------------------------------------
  # 2. Reproducibility
  # ----------------------------------------------------------

  try(
    keras3::clear_session(
      free_memory = FALSE
    ),
    silent = TRUE
  )


  set.seed(
    as.integer(seed)
  )


  keras3::set_random_seed(
    as.integer(seed)
  )


  # ----------------------------------------------------------
  # 3. Build matched network
  # ----------------------------------------------------------

  model <-
    build_nn(

      input_dim =
        ncol(X_train),

      depth =
        as.integer(depth),

      width =
        as.integer(width),

      lambda_nn =
        as.numeric(lambda_nn),

      penalty_tau =
        as.numeric(penalty_tau),

      learning_rate =
        as.numeric(learning_rate)
    )


  # ----------------------------------------------------------
  # 4. Early stopping
  #
  # Monitor the predictive BCE metric rather than Keras
  # val_loss, since val_loss can include regularisation terms.
  # ----------------------------------------------------------

  callback <-
    keras3::callback_early_stopping(

      monitor =
        "val_binary_crossentropy",

      mode =
        "min",

      patience =
        as.integer(patience),

      restore_best_weights =
        TRUE
    )


  # ----------------------------------------------------------
  # 5. Fit
  # ----------------------------------------------------------

  # ----------------------------------------------------------
# Fit through the R keras3 S3 interface.
#
# IMPORTANT:
# Using model$fit() returns the underlying Python History
# object.  generics::fit() dispatches to the keras3 R method
# and returns a keras_training_history object containing
# $metrics.
# ----------------------------------------------------------

history <-
  generics::fit(

    model,

    x =
      X_train,

    y =
      y_array,

    epochs =
      as.integer(epochs),

    batch_size =
      as.integer(batch_size),

    validation_split =
      as.numeric(validation_split),

    shuffle =
      TRUE,

    callbacks =
      list(
        callback
      ),

    verbose =
      if (verbose) 1L else 0L
  )

  # ----------------------------------------------------------
  # 6. Extract optimisation diagnostics
  # ----------------------------------------------------------

  history_metrics <-
    history$metrics


  if (
    !is.null(
      history_metrics$val_binary_crossentropy
    )
  ) {

    val_bce <-
      as.numeric(
        history_metrics$val_binary_crossentropy
      )

  } else if (
    !is.null(
      history_metrics$val_loss
    )
  ) {

    # Defensive fallback only.
    val_bce <-
      as.numeric(
        history_metrics$val_loss
      )

  } else {

    val_bce <-
      numeric(0)
  }


  epochs_used <-
    if (
      length(val_bce) > 0L
    ) {

      length(val_bce)

    } else {

      NA_integer_
    }


  best_epoch <-
    if (
      length(val_bce) > 0L &&
      any(is.finite(val_bce))
    ) {

      which.min(val_bce)

    } else {

      NA_integer_
    }


  best_inner_val_bce <-
    if (
      is.finite(best_epoch)
    ) {

      val_bce[best_epoch]

    } else {

      NA_real_
    }


  # ----------------------------------------------------------
  # 7. Independent prediction
  # ----------------------------------------------------------

  prob <-
    as.numeric(
      model(
        X_test,
        training = FALSE
      )
    )


  prob <-
    pmin(
      pmax(
        prob,
        0
      ),
      1
    )


  pred <-
    as.integer(
      prob >= 0.5
    )


  # ----------------------------------------------------------
  # 8. Return
  # ----------------------------------------------------------

  list(

    pred =
      pred,

    prob =
      prob,

    epochs_used =
      epochs_used,

    best_epoch =
      best_epoch,

    best_inner_val_bce =
      best_inner_val_bce,

    model =
      model,

    history =
      history
  )
}




evaluate_predictions <- function(
    y_true,
    pred = NULL,
    prob,
    true_prob,
    threshold = 0.5
) {

  # ----------------------------------------------------------
  # Convert inputs
  # ----------------------------------------------------------

  y_true <-
    as.integer(
      y_true
    )


  prob <-
    as.numeric(
      prob
    )


  true_prob <-
    as.numeric(
      true_prob
    )


  # ----------------------------------------------------------
  # Basic validation
  # ----------------------------------------------------------

  n <-
    length(
      y_true
    )


  if (
    length(prob) != n ||
    length(true_prob) != n
  ) {

    stop(
      "y_true, prob and true_prob must have the same length.",
      call. = FALSE
    )
  }


  if (
    any(
      !y_true %in% c(
        0L,
        1L
      )
    )
  ) {

    stop(
      "y_true must contain only 0/1 outcomes.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        prob
      )
    ) ||
    any(
      !is.finite(
        true_prob
      )
    )
  ) {

    stop(
      "prob and true_prob must contain only finite values.",
      call. = FALSE
    )
  }


  if (
    any(
      prob < 0 |
        prob > 1
    )
  ) {

    stop(
      "prob must lie in [0, 1].",
      call. = FALSE
    )
  }


  if (
    any(
      true_prob < 0 |
        true_prob > 1
    )
  ) {

    stop(
      "true_prob must lie in [0, 1].",
      call. = FALSE
    )
  }


  if (
    length(threshold) != 1L ||
    !is.finite(threshold) ||
    threshold <= 0 ||
    threshold >= 1
  ) {

    stop(
      "threshold must lie strictly between 0 and 1.",
      call. = FALSE
    )
  }


  # ----------------------------------------------------------
  # Classification rule
  #
  # Reconstruct from fitted probabilities so that probability
  # and classification metrics are internally consistent.
  # ----------------------------------------------------------

  pred_from_prob <-
    as.integer(
      prob >= threshold
    )


  if (
    !is.null(pred)
  ) {

    pred <-
      as.integer(
        pred
      )


    if (
      length(pred) != n
    ) {

      stop(
        "pred must have the same length as y_true.",
        call. = FALSE
      )
    }


    if (
      any(
        !pred %in% c(
          0L,
          1L
        )
      )
    ) {

      stop(
        "pred must contain only 0/1 values.",
        call. = FALSE
      )
    }


    if (
      any(
        pred != pred_from_prob
      )
    ) {

      warning(
        "Provided pred differs from thresholded prob; thresholded probabilities are used.",
        call. = FALSE
      )
    }
  }


  pred <-
    pred_from_prob


  # ----------------------------------------------------------
  # Numerical clipping for logarithmic losses only
  # ----------------------------------------------------------

  eps <-
    1e-7


  prob_log <-
    pmin(
      pmax(
        prob,
        eps
      ),
      1 - eps
    )


  true_prob_log <-
    pmin(
      pmax(
        true_prob,
        eps
      ),
      1 - eps
    )


  # ==========================================================
  # EMPIRICAL TEST-SAMPLE METRICS
  # ==========================================================


  # ----------------------------------------------------------
  # Observed misclassification risk
  #
  #   n^{-1} sum I(g_hat(X_i) != Y_i)
  # ----------------------------------------------------------

  risk <-
    mean(
      pred !=
        y_true
    )


  # ----------------------------------------------------------
  # Observed logistic loss
  # ----------------------------------------------------------

  empirical_logistic_loss <-
    -mean(

      y_true *
        log(
          prob_log
        ) +

        (
          1 -
            y_true
        ) *
        log(
          1 -
            prob_log
        )
    )


  # ----------------------------------------------------------
  # Observed Brier score
  # ----------------------------------------------------------

  brier <-
    mean(
      (
        prob -
          y_true
      )^2
    )


  # ----------------------------------------------------------
  # Empirical AUC
  # ----------------------------------------------------------

  auc <-
    safe_auc(

      y =
        y_true,

      prob =
        prob
    )


  # ==========================================================
  # CONDITIONAL POPULATION METRICS
  # ==========================================================


  # ----------------------------------------------------------
  # Conditional classification risk
  #
  # For each test covariate X_i,
  #
  #   R_i(g_hat)
  #     =
  #     pi_i        if g_hat(X_i) = 0,
  #     1 - pi_i    if g_hat(X_i) = 1.
  # ----------------------------------------------------------

  conditional_risk <-
    mean(

      ifelse(

        pred == 1L,

        1 -
          true_prob,

        true_prob
      )
    )


  # ----------------------------------------------------------
  # Bayes classification risk
  #
  #   R* = E[min{pi(X), 1-pi(X)}]
  # ----------------------------------------------------------

  bayes_class_risk <-
    mean(

      pmin(

        true_prob,

        1 -
          true_prob
      )
    )


  # ----------------------------------------------------------
  # Excess classification risk
  # ----------------------------------------------------------

  excess_class_risk <-
    conditional_risk -
    bayes_class_risk


  # ----------------------------------------------------------
  # Conditional logistic risk of fitted probabilities
  # ----------------------------------------------------------

  logistic_risk <-
    -mean(

      true_prob *
        log(
          prob_log
        ) +

        (
          1 -
            true_prob
        ) *
        log(
          1 -
            prob_log
        )
    )


  # ----------------------------------------------------------
  # Bayes logistic risk
  #
  # Achieved by predicting the true conditional probability.
  # ----------------------------------------------------------

  bayes_logistic_risk <-
    -mean(

      true_prob *
        log(
          true_prob_log
        ) +

        (
          1 -
            true_prob
        ) *
        log(
          1 -
            true_prob_log
        )
    )


  # ----------------------------------------------------------
  # Surrogate excess risk
  # ----------------------------------------------------------

  surrogate_excess_risk <-
    logistic_risk -
    bayes_logistic_risk


  # ----------------------------------------------------------
  # Conditional population Brier risk
  #
  # E[(Y - p_hat)^2 | X]
  #
  #   =
  #   pi(1-pi)
  #   +
  #   (p_hat-pi)^2.
  # ----------------------------------------------------------

  population_brier <-
    mean(

      true_prob *
        (
          1 -
            true_prob
        ) +

        (
          prob -
            true_prob
        )^2
    )


  # ----------------------------------------------------------
  # Bayes Brier risk
  #
  # Attained at p_hat = pi(X).
  # ----------------------------------------------------------

  bayes_brier <-
    mean(

      true_prob *
        (
          1 -
            true_prob
        )
    )


  # ----------------------------------------------------------
  # Excess Brier risk
  # ----------------------------------------------------------

  excess_brier <-
    population_brier -
    bayes_brier


  # ==========================================================
  # Return
  # ==========================================================

  tibble::tibble(

    risk =
      risk,

    conditional_risk =
      conditional_risk,

    bayes_class_risk =
      bayes_class_risk,

    excess_class_risk =
      excess_class_risk,

    empirical_logistic_loss =
      empirical_logistic_loss,

    logistic_risk =
      logistic_risk,

    bayes_logistic_risk =
      bayes_logistic_risk,

    surrogate_excess_risk =
      surrogate_excess_risk,

    auc =
      auc,

    brier =
      brier,

    population_brier =
      population_brier,

    bayes_brier =
      bayes_brier,

    excess_brier =
      excess_brier
  )
}

binary_log_loss <- function(
    y,
    prob,
    eps = 1e-7
) {

  y <-
    as.numeric(
      y
    )


  prob <-
    as.numeric(
      prob
    )


  if (
    length(y) !=
    length(prob)
  ) {

    stop(
      "y and prob must have the same length.",
      call. = FALSE
    )
  }


  if (
    any(
      !y %in%
      c(
        0,
        1
      )
    )
  ) {

    stop(
      "y must contain only 0/1 outcomes.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        prob
      )
    )
  ) {

    stop(
      "prob must contain only finite values.",
      call. = FALSE
    )
  }


  prob <-
    pmin(
      pmax(
        prob,
        eps
      ),
      1 -
      eps
    )


  -mean(

    y *
      log(
        prob
      ) +

      (
        1 -
          y
      ) *
      log(
        1 -
          prob
      )
  )
}

make_stratified_tuning_split <- function(
    y,
    validation_fraction = 0.20,
    seed = 1L
) {

  y <-
    as.integer(
      y
    )


  if (
    any(
      !y %in%
      c(
        0L,
        1L
      )
    )
  ) {

    stop(
      "y must contain only 0/1 outcomes.",
      call. = FALSE
    )
  }


  if (
    length(
      unique(
        y
      )
    ) < 2L
  ) {

    stop(
      "Both outcome classes are required.",
      call. = FALSE
    )
  }


  if (
    length(validation_fraction) != 1L ||
    !is.finite(validation_fraction) ||
    validation_fraction <= 0 ||
    validation_fraction >= 0.5
  ) {

    stop(
      "validation_fraction must lie strictly between 0 and 0.5.",
      call. = FALSE
    )
  }


  set.seed(
    seed
  )


  idx0 <-
    which(
      y == 0L
    )


  idx1 <-
    which(
      y == 1L
    )


  if (
    length(idx0) < 2L ||
    length(idx1) < 2L
  ) {

    stop(
      "Each class must contain at least two observations.",
      call. = FALSE
    )
  }


  n_val0 <-
    max(
      1L,
      floor(
        validation_fraction *
          length(
            idx0
          )
      )
    )


  n_val1 <-
    max(
      1L,
      floor(
        validation_fraction *
          length(
            idx1
          )
      )
    )


  # Ensure at least one observation from each class remains
  # in the fitting sample.

  n_val0 <-
    min(
      n_val0,
      length(idx0) -
        1L
    )


  n_val1 <-
    min(
      n_val1,
      length(idx1) -
        1L
    )


  val_idx <-
    sort(
      c(

        sample(
          idx0,
          size =
            n_val0,
          replace =
            FALSE
        ),

        sample(
          idx1,
          size =
            n_val1,
          replace =
            FALSE
        )
      )
    )


  fit_idx <-
    setdiff(
      seq_along(
        y
      ),
      val_idx
    )


  list(

    fit_idx =
      fit_idx,

    val_idx =
      val_idx,

    validation_fraction =
      validation_fraction
  )
}

fit_nn_fixed_validation <- function(
    X_fit,
    y_fit,
    X_val,
    y_val,
    depth,
    width,
    lambda_nn,
    penalty_tau = 1,
    learning_rate = 1e-3,
    epochs = 100L,
    batch_size = 32L,
    patience = 10L,
    seed = 1L
) {

  X_fit <-
    as.matrix(
      X_fit
    )


  X_val <-
    as.matrix(
      X_val
    )


  y_fit <-
    as.numeric(
      y_fit
    )


  y_val <-
    as.numeric(
      y_val
    )


  if (
    nrow(X_fit) !=
    length(y_fit)
  ) {

    stop(
      "nrow(X_fit) must equal length(y_fit).",
      call. = FALSE
    )
  }


  if (
    nrow(X_val) !=
    length(y_val)
  ) {

    stop(
      "nrow(X_val) must equal length(y_val).",
      call. = FALSE
    )
  }


  if (
    ncol(X_fit) !=
    ncol(X_val)
  ) {

    stop(
      "X_fit and X_val must contain the same predictors.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        X_fit
      )
    ) ||
    any(
      !is.finite(
        X_val
      )
    )
  ) {

    stop(
      "Predictor matrices must contain only finite values.",
      call. = FALSE
    )
  }


  y_fit_array <-
    array(
      y_fit,
      dim =
        c(
          length(y_fit),
          1L
        )
    )


  y_val_array <-
    array(
      y_val,
      dim =
        c(
          length(y_val),
          1L
        )
    )


  storage.mode(
    X_fit
  ) <- "double"


  storage.mode(
    X_val
  ) <- "double"


  storage.mode(
    y_fit_array
  ) <- "double"


  storage.mode(
    y_val_array
  ) <- "double"


  try(
    keras3::clear_session(),
    silent =
      TRUE
  )


  set_model_seed(
    seed
  )


  model <-
    build_nn(

      input_dim =
        ncol(
          X_fit
        ),

      depth =
        depth,

      width =
        width,

      lambda_nn =
        lambda_nn,

      penalty_tau =
        penalty_tau,

      learning_rate =
        learning_rate
    )


  early_stop <-
    keras3::callback_early_stopping(

      monitor =
  "val_binary_crossentropy",

      patience =
        as.integer(
          patience
        ),

      restore_best_weights =
        TRUE
    )


  history <-
    model |>
    fit(

      x =
        X_fit,

      y =
        y_fit_array,

      validation_data =
        list(
          X_val,
          y_val_array
        ),

      epochs =
        as.integer(
          epochs
        ),

      batch_size =
        as.integer(
          batch_size
        ),

      callbacks =
        list(
          early_stop
        ),

      verbose =
        0
    )


  val_prob <-
    as.numeric(

      model |>
      predict(
        X_val,
        verbose =
          0
      )
    )


  val_loss <-
    binary_log_loss(

      y =
        y_val,

      prob =
        val_prob
    )


  # ----------------------------------------------------------
  # Validation diagnostics
  # ----------------------------------------------------------

  validation_auc <-
    safe_auc(
      y =
        y_val,
      prob =
        val_prob
    )


  validation_brier <-
    mean(
      (
        val_prob -
        y_val
      )^2
    )


  null_probability <-
    mean(
      y_fit
    )


  null_validation_loss <-
    binary_log_loss(

      y =
        y_val,

      prob =
        rep(
          null_probability,
          length(
            y_val
          )
        )
    )


  validation_loss_gain <-
    null_validation_loss -
    val_loss


  validation_prob_sd <-
    sd(
      val_prob
    )


  validation_prob_min <-
    min(
      val_prob
    )


  validation_prob_max <-
    max(
      val_prob
    )


  # ----------------------------------------------------------
  # Training-history diagnostics
  #
  # Hyperparameter selection and early stopping are based on
  # predictive binary cross-entropy rather than the penalised
  # Keras loss.
  # ----------------------------------------------------------

  bce_history <-
    as.numeric(
      unlist(
        history$metrics[["binary_crossentropy"]]
      )
    )


  val_bce_history <-
    as.numeric(
      unlist(
        history$metrics[["val_binary_crossentropy"]]
      )
    )


  if (
    length(bce_history) == 0L ||
    length(val_bce_history) == 0L
  ) {

    stop(
      paste0(
        "Keras training history does not contain ",
        "binary_crossentropy/val_binary_crossentropy."
      ),
      call. = FALSE
    )
  }


  epochs_used <-
    length(
      bce_history
    )


  best_epoch <-
    which.min(
      val_bce_history
    )


  best_history_val_bce <-
    min(
      val_bce_history,
      na.rm = TRUE
    )

  return(
  list(

    model =
      model,

    val_prob =
      val_prob,

    val_loss =
      val_loss,

    validation_auc =
      validation_auc,

    validation_brier =
      validation_brier,

    null_validation_loss =
      null_validation_loss,

    validation_loss_gain =
      validation_loss_gain,

    validation_prob_sd =
      validation_prob_sd,

    validation_prob_min =
      validation_prob_min,

    validation_prob_max =
      validation_prob_max,

    epochs_used =
      epochs_used,

    best_epoch =
      best_epoch,

    best_history_val_bce =
  best_history_val_bce
  )
)
}



tune_crdnn_cv <- function(
    X_train,
    y_train,
    depth,
    width,
    alpha_grid = c(
      0.01,
      0.025,
      0.05,
      0.10,
      0.25,
      0.50,
      0.75,
      1.00
    ),
    lambda_nn_grid = c(
      0,
      1e-7,
      1e-6,
      1e-5,
      1e-4,
      3e-4,
      1e-3,
      3e-3,
      1e-2
    ),
    foldid = NULL,
    nfolds = 5L,
    inner_validation_fraction = 0.10,
    penalty_tau = 1,
    learning_rate = 1e-3,
    epochs = 100L,
    batch_size = 32L,
    patience = 10L,
    seed = 1L
) {

  # ==========================================================
  # 11.7.1 Input preparation
  # ==========================================================

  X_train <-
    as.matrix(
      X_train
    )


  y_train <-
    as.integer(
      y_train
    )


  n <-
    nrow(
      X_train
    )


  if (
    length(y_train) != n
  ) {

    stop(
      "nrow(X_train) must equal length(y_train).",
      call. = FALSE
    )
  }


  if (
    any(
      !y_train %in%
      c(
        0L,
        1L
      )
    )
  ) {

    stop(
      "y_train must contain only 0/1 outcomes.",
      call. = FALSE
    )
  }


  if (
    length(
      unique(
        y_train
      )
    ) < 2L
  ) {

    stop(
      "Both outcome classes are required.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        X_train
      )
    )
  ) {

    stop(
      "X_train contains non-finite values.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        alpha_grid
      )
    ) ||
    any(
      alpha_grid <= 0 |
      alpha_grid > 1
    )
  ) {

    stop(
      "alpha_grid must satisfy 0 < alpha <= 1.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        lambda_nn_grid
      )
    ) ||
    any(
      lambda_nn_grid < 0
    )
  ) {

    stop(
      "lambda_nn_grid must contain finite non-negative values.",
      call. = FALSE
    )
  }


  alpha_grid <-
    sort(
      unique(
        alpha_grid
      )
    )


  lambda_nn_grid <-
    sort(
      unique(
        lambda_nn_grid
      )
    )


  # ==========================================================
  # 11.7.2 Common stratified outer folds
  # ==========================================================

  if (
    is.null(
      foldid
    )
  ) {

    foldid <-
      make_stratified_foldid(

        y =
          y_train,

        k =
          nfolds,

        seed =
          seed
      )
  }


  if (
    length(foldid) != n
  ) {

    stop(
      "foldid must contain one entry per training observation.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        foldid
      )
    )
  ) {

    stop(
      "foldid contains non-finite values.",
      call. = FALSE
    )
  }


  foldid <-
    as.integer(
      foldid
    )


  fold_levels <-
    sort(
      unique(
        foldid
      )
    )


  K <-
    length(
      fold_levels
    )


  if (
    K < 3L
  ) {

    stop(
      "At least three outer folds are required.",
      call. = FALSE
    )
  }


  # ----------------------------------------------------------
  # Ensure each outer fitting sample contains both classes
  # ----------------------------------------------------------

  for (
    fold_k in fold_levels
  ) {

    fit_idx_check <-
      which(
        foldid !=
          fold_k
      )


    if (
      length(
        unique(
          y_train[
            fit_idx_check
          ]
        )
      ) < 2L
    ) {

      stop(
        "Each outer fitting fold must contain both outcome classes.",
        call. = FALSE
      )
    }
  }


  # ==========================================================
  # 11.7.3 Candidate grid and storage
  # ==========================================================

  candidate_grid <-
    tidyr::expand_grid(

      alpha =
        alpha_grid,

      lambda_nn =
        lambda_nn_grid
    ) |>

    dplyr::mutate(

      candidate_id =
        dplyr::row_number()
    )


  n_candidates <-
    nrow(
      candidate_grid
    )


  oof_prob <-
    matrix(

      NA_real_,

      nrow =
        n,

      ncol =
        n_candidates
    )


  oof_null_prob <-
    rep(
      NA_real_,
      n
    )


  epochs_matrix <-
    matrix(

      NA_real_,

      nrow =
        K,

      ncol =
        n_candidates
    )


  best_epoch_matrix <-
    matrix(

      NA_real_,

      nrow =
        K,

      ncol =
        n_candidates
    )


  condition_matrix <-
    matrix(

      NA_real_,

      nrow =
        K,

      ncol =
        length(
          alpha_grid
        )
    )


  # ==========================================================
  # 11.7.4 Outer CV loop
  # ==========================================================

  for (
    kk in seq_along(
      fold_levels
    )
  ) {

    fold_k <-
      fold_levels[kk]


    val_idx <-
      which(
        foldid ==
          fold_k
      )


    fit_idx <-
      which(
        foldid !=
          fold_k
      )


    X_fit_raw <-
      X_train[
        fit_idx,
        ,
        drop = FALSE
      ]


    X_val_raw <-
      X_train[
        val_idx,
        ,
        drop = FALSE
      ]


    y_fit <-
      y_train[
        fit_idx
      ]


    # --------------------------------------------------------
    # Training-fold null prediction
    # --------------------------------------------------------

    oof_null_prob[
      val_idx
    ] <-
      mean(
        y_fit
      )


    # --------------------------------------------------------
    # Shrinkage preparation ONCE per outer fold
    # --------------------------------------------------------

    shrink_prep <-
      prepare_shrinkage(
        X_fit_raw
      )


    # --------------------------------------------------------
    # Matched randomisation across all candidates in fold
    # --------------------------------------------------------

    fold_seed <-
      seed +
      10000L *
      kk


    # ========================================================
    # Alpha loop
    # ========================================================

    for (
      aa in seq_along(
        alpha_grid
      )
    ) {

      alpha_a <-
        alpha_grid[aa]


      fit_white <-
        apply_shrinkage(

          prep =
            shrink_prep,

          X =
            X_fit_raw,

          alpha =
            alpha_a
        )


      val_white <-
        apply_shrinkage(

          prep =
            shrink_prep,

          X =
            X_val_raw,

          alpha =
            alpha_a
        )


      condition_matrix[
        kk,
        aa
      ] <-
        fit_white$condition_number


      # ======================================================
      # Lambda loop
      # ======================================================

      for (
        ll in seq_along(
          lambda_nn_grid
        )
      ) {

        lambda_l <-
          lambda_nn_grid[ll]


        candidate_id <-
          candidate_grid |>

          dplyr::filter(

            alpha ==
              alpha_a,

            lambda_nn ==
              lambda_l
          ) |>

          dplyr::pull(
            candidate_id
          )


        if (
          length(candidate_id) != 1L
        ) {

          stop(
            "Could not identify a unique CR-DNN tuning candidate.",
            call. = FALSE
          )
        }


        fit_l <-
          fit_predict_nn(

            X_train =
              fit_white$Z,

            y_train =
              y_fit,

            X_test =
              val_white$Z,

            depth =
              depth,

            width =
              width,

            lambda_nn =
              lambda_l,

            penalty_tau =
              penalty_tau,

            learning_rate =
              learning_rate,

            epochs =
              epochs,

            batch_size =
              batch_size,

            validation_split =
              inner_validation_fraction,

            patience =
              patience,

            seed =
              fold_seed,

            verbose =
              FALSE
          )


        oof_prob[
          val_idx,
          candidate_id
        ] <-
          fit_l$prob


        epochs_matrix[
          kk,
          candidate_id
        ] <-
          fit_l$epochs_used


        best_epoch_matrix[
          kk,
          candidate_id
        ] <-
          fit_l$best_epoch
      }
    }
  }


  # ==========================================================
  # 11.7.5 Verify complete OOF predictions
  # ==========================================================

  if (
    any(
      !is.finite(
        oof_prob
      )
    )
  ) {

    stop(
      "Incomplete CR-DNN out-of-fold predictions.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        oof_null_prob
      )
    )
  ) {

    stop(
      "Incomplete null-model out-of-fold predictions.",
      call. = FALSE
    )
  }


  # ==========================================================
  # 11.7.6 Pooled CV metrics
  # ==========================================================

  null_cv_loss <-
    binary_log_loss(

      y =
        y_train,

      prob =
        oof_null_prob
    )


  tuning_results <-
    vector(

      mode =
        "list",

      length =
        n_candidates
    )


  for (
    cc in seq_len(
      n_candidates
    )
  ) {

    alpha_c <-
      candidate_grid$alpha[cc]


    lambda_c <-
      candidate_grid$lambda_nn[cc]


    prob_c <-
      oof_prob[
        ,
        cc
      ]


    alpha_index <-
      match(
        alpha_c,
        alpha_grid
      )


    if (
      is.na(
        alpha_index
      )
    ) {

      stop(
        "Could not match candidate alpha to alpha_grid.",
        call. = FALSE
      )
    }


    cv_loss <-
      binary_log_loss(

        y =
          y_train,

        prob =
          prob_c
      )


    tuning_results[[cc]] <-
      tibble::tibble(

        alpha =
          alpha_c,

        lambda_nn =
          lambda_c,

        cv_validation_loss =
          cv_loss,

        null_cv_loss =
          null_cv_loss,

        cv_loss_gain =
          null_cv_loss -
          cv_loss,

        cv_auc =
          safe_auc(

            y =
              y_train,

            prob =
              prob_c
          ),

        cv_brier =
          mean(
            (
              prob_c -
                y_train
            )^2
          ),

        cv_prob_sd =
          sd(
            prob_c
          ),

        cv_prob_min =
          min(
            prob_c
          ),

        cv_prob_max =
          max(
            prob_c
          ),

        mean_epochs_used =
          mean(
            epochs_matrix[
              ,
              cc
            ],
            na.rm =
              TRUE
          ),

        mean_best_epoch =
          mean(
            best_epoch_matrix[
              ,
              cc
            ],
            na.rm =
              TRUE
          ),

        mean_condition_number =
          mean(
            condition_matrix[
              ,
              alpha_index
            ],
            na.rm =
              TRUE
          ),

        outer_folds =
          K
      )
  }


  tuning_table <-
    dplyr::bind_rows(
      tuning_results
    ) |>

    dplyr::arrange(
      cv_validation_loss,
      alpha,
      lambda_nn
    )


  selected_alpha <-
    tuning_table$alpha[1L]


  selected_lambda <-
    tuning_table$lambda_nn[1L]


  # ==========================================================
  # 11.7.7 Return
  # ==========================================================

  return(
    list(

      selected_alpha =
        selected_alpha,

      selected_lambda =
        selected_lambda,

      tuning_table =
        tuning_table,

      foldid =
        foldid,

      candidate_grid =
        candidate_grid,

      oof_prob =
        oof_prob,

      oof_null_prob =
        oof_null_prob
    )
  )
}

make_stratified_calibration_split <- function(
    y,
    validation_fraction = 0.10,
    seed = 72026L
) {

  y <-
    as.integer(
      y
    )

  if (
    !all(
      sort(unique(y)) ==
      c(0L, 1L)
    )
  ) {

    stop(
      "Binary classes 0 and 1 are required.",
      call. = FALSE
    )
  }


  set.seed(
    seed
  )


  validation_index <-
    integer(0)


  for (
    cls in c(0L, 1L)
  ) {

    idx <-
      which(
        y == cls
      )


    n_validation <-
      max(
        1L,
        as.integer(
          round(
            validation_fraction *
            length(idx)
          )
        )
      )


    n_validation <-
      min(
        n_validation,
        length(idx) - 1L
      )


    validation_index <-
      c(
        validation_index,
        sample(
          idx,
          size =
            n_validation,
          replace =
            FALSE
        )
      )
  }


  validation_index <-
    sort(
      validation_index
    )


  fitting_index <-
    setdiff(
      seq_along(y),
      validation_index
    )


  if (
    length(
      unique(
        y[fitting_index]
      )
    ) < 2L ||
    length(
      unique(
        y[validation_index]
      )
    ) < 2L
  ) {

    stop(
      "Both calibration subsets must contain both classes.",
      call. = FALSE
    )
  }


  return(
    list(

      fitting_index =
        fitting_index,

      validation_index =
        validation_index
    )
  )
}

fit_predict_nn_final <- function(
    X_fit,
    y_fit,
    X_validation,
    y_validation,
    X_test,
    depth,
    width,
    lambda_nn,
    penalty_tau = 1,
    learning_rate = 1e-3,
    epochs = 100L,
    batch_size = 32L,
    patience = 10L,
    seed = 1L,
    verbose = FALSE
) {

  X_fit <-
    as.matrix(X_fit)

  X_validation <-
    as.matrix(X_validation)

  X_test <-
    as.matrix(X_test)

  y_fit <-
    as.integer(y_fit)

  y_validation <-
    as.integer(y_validation)


  keras3::clear_session(
    free_memory = FALSE
  )


  keras3::set_random_seed(
    as.integer(seed)
  )


  model <-
    build_nn(

      input_dim =
        ncol(X_fit),

      depth =
        depth,

      width =
        width,

      lambda_nn =
        lambda_nn,

      penalty_tau =
        penalty_tau,

      learning_rate =
        learning_rate
    )


  y_fit_array <-
    array(
      as.numeric(y_fit),
      dim =
        c(
          length(y_fit),
          1L
        )
    )


  y_validation_array <-
    array(
      as.numeric(y_validation),
      dim =
        c(
          length(y_validation),
          1L
        )
    )


  history <-
    model |>
    fit(

      x =
        X_fit,

      y =
        y_fit_array,

      validation_data =
        list(
          X_validation,
          y_validation_array
        ),

      epochs =
        as.integer(epochs),

      batch_size =
        as.integer(batch_size),

      callbacks =
        list(

          keras3::callback_early_stopping(

            monitor =
              "val_binary_crossentropy",

            mode =
              "min",

            patience =
              as.integer(patience),

            restore_best_weights =
              TRUE
          )
        ),

      verbose =
        as.integer(verbose)
    )


  val_bce <-
    history$metrics$
    val_binary_crossentropy


  if (
    is.null(val_bce) ||
    length(val_bce) == 0L
  ) {

    stop(
      "val_binary_crossentropy was not recorded.",
      call. = FALSE
    )
  }


  prob <-
    as.numeric(
      as.array(
        model(
          X_test,
          training = FALSE
        )
      )
    )


  pred <-
    as.integer(
      prob >= 0.5
    )


  return(
    list(

      pred =
        pred,

      prob =
        prob,

      model =
        model,

      best_epoch =
        as.integer(
          which.min(val_bce)
        ),

      epochs_used =
        as.integer(
          length(val_bce)
        ),

      best_validation_bce =
        min(val_bce)
    )
  )
}

# ============================================================
# 12. FROZEN PRODUCTION SPECIFICATION
# ============================================================

final_simulation_spec <- list(

  scenarios = c(
    "near_iso_nonlinear",
    "moderate_aniso_nonlinear",
    "strong_aniso_nonlinear",
    "near_iso_linear",
    "moderate_aniso_linear",
    "strong_aniso_linear"
  ),

  n = 1000L,

  p_over_n = c(
    0.5,
    1.0,
    1.5
  ),

  depth = c(
    1L,
    3L,
    5L
  ),

  B = 20L,

  n_test = 5000L,

  # Frozen CR-DNN covariance-shrinkage grid.
  # alpha = 0.05 is now an interior candidate.
  alpha_grid = c(
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00
  ),

  # Preserve the original grid used for the completed shrinkage-LDA
  # benchmark; sequential production does not rerun the linear layer.
  lda_alpha_grid = c(
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00
  ),

  lambda_nn_grid = c(
    0,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2
  ),

  enet_mixing_grid = c(
    0.25,
    0.50,
    0.75
  ),

  hidden_width = 32L,
  penalty_tau = 1,
  learning_rate = 1e-3,
  max_epochs = 100L,
  batch_size = 32L,
  patience = 10L,
  inner_validation_fraction = 0.10,
  outer_folds = 5L,
  final_validation_fraction = 0.10,
  master_seed = 260000L
)


get_final_signal_multiplier <- function(
    scenario
) {

  spec <-
    get_scenario_spec(
      scenario
    )

  as.numeric(
    spec$signal_multiplier
  )
}


network_method_label <- function(
    depth
) {

  depth <- as.integer(depth)

  if (depth == 1L) {
    "ANN"
  } else {
    "DNN"
  }
}


# ------------------------------------------------------------
# Common random-number seed map.
#
# The structural and sampled-data seeds depend on p and
# replication, but NOT on scenario or network depth.  Thus:
#
#   - linear/nonlinear response scenarios use common base
#     predictor random numbers;
#   - covariance regimes use common Gaussian innovations;
#   - depths use exactly the same training/test data.
# ------------------------------------------------------------

get_structural_seed <- function(
    p,
    master_seed =
      final_simulation_spec$master_seed
) {

  as.integer(
    master_seed +
    100000L +
    as.integer(p)
  )
}


get_replication_seed <- function(
    p,
    rep,
    master_seed =
      final_simulation_spec$master_seed
) {

  as.integer(
    master_seed +
    1000000L +
    10000L *
    as.integer(p) +
    100L *
    as.integer(rep)
  )
}


# ============================================================
# 13. LINEAR BENCHMARK FUNCTIONS
#
# These methods are depth independent and are therefore fitted
# once per scenario x p/n x replication:
#
#   - Shrinkage LDA
#   - Ridge logistic regression
#   - Elastic-net logistic regression
# ============================================================


# ============================================================
# 13.1 SHRINKAGE LDA PREPARATION
#
# The pooled within-class covariance is estimated using
#
#   Sigma_W_hat
#     = n^{-1} sum_k sum_{i:Y_i=k}
#         (X_i-mu_k)(X_i-mu_k)^T.
#
# The same shrinkage family used conceptually by CR-DNN is
#
#   Sigma_W_tilde(alpha)
#     = (1-alpha) Sigma_W_hat
#       + alpha tau_W I_p.
# ============================================================

prepare_shrinkage_lda <- function(
    X_train,
    y_train
) {

  X_train <- as.matrix(X_train)
  y_train <- as.integer(y_train)

  if (nrow(X_train) != length(y_train)) {
    stop(
      "X_train and y_train have incompatible dimensions.",
      call. = FALSE
    )
  }

  if (
    !identical(
      sort(unique(y_train)),
      c(0L, 1L)
    )
  ) {
    stop(
      "Shrinkage LDA requires both binary classes coded 0/1.",
      call. = FALSE
    )
  }

  if (any(!is.finite(X_train))) {
    stop(
      "X_train contains non-finite values.",
      call. = FALSE
    )
  }

  n <- nrow(X_train)
  p <- ncol(X_train)

  idx0 <- which(y_train == 0L)
  idx1 <- which(y_train == 1L)

  mu0 <-
    colMeans(
      X_train[
        idx0,
        ,
        drop = FALSE
      ]
    )

  mu1 <-
    colMeans(
      X_train[
        idx1,
        ,
        drop = FALSE
      ]
    )

  prior0 <- length(idx0) / n
  prior1 <- length(idx1) / n

  Xw <-
    matrix(
      0,
      nrow = n,
      ncol = p
    )

  Xw[idx0, ] <-
    sweep(
      X_train[
        idx0,
        ,
        drop = FALSE
      ],
      2,
      mu0,
      FUN = "-"
    )

  Xw[idx1, ] <-
    sweep(
      X_train[
        idx1,
        ,
        drop = FALSE
      ],
      2,
      mu1,
      FUN = "-"
    )

  tau_cov <-
    sum(Xw^2) /
    (n * p)

  if (!is.finite(tau_cov) || tau_cov <= 0) {
    stop(
      "Invalid pooled within-class variance target.",
      call. = FALSE
    )
  }

  ss <-
    svd(
      Xw,
      nu = 0L,
      nv = min(n, p)
    )

  if (length(ss$d) == 0L) {

    V <-
      matrix(
        numeric(0),
        nrow = p,
        ncol = 0L
      )

    eigvals <- numeric(0)

  } else {

    tol <-
      max(dim(Xw)) *
      max(ss$d) *
      .Machine$double.eps

    keep <-
      ss$d >
      tol

    eigvals <-
      ss$d[keep]^2 /
      n

    V <-
      ss$v[
        ,
        keep,
        drop = FALSE
      ]
  }

  list(
    mu0 = mu0,
    mu1 = mu1,
    prior0 = prior0,
    prior1 = prior1,
    V = V,
    eigvals = eigvals,
    tau_cov = tau_cov,
    n = n,
    p = p
  )
}


# ============================================================
# 13.2 APPLY INVERSE SHRINKAGE COVARIANCE TO A VECTOR
# ============================================================

solve_shrinkage_covariance <- function(
    prep,
    v,
    alpha
) {

  v <- as.numeric(v)
  alpha <- as.numeric(alpha)

  if (
    length(v) != prep$p ||
    length(alpha) != 1L ||
    !is.finite(alpha) ||
    alpha <= 0 ||
    alpha > 1
  ) {
    stop(
      "Invalid vector or shrinkage parameter.",
      call. = FALSE
    )
  }

  c0 <-
    alpha *
    prep$tau_cov

  out <-
    v /
    c0

  if (
    length(
      prep$eigvals
    ) > 0L
  ) {

    projected <-
      drop(
        crossprod(
          prep$V,
          v
        )
      )

    shrunk <-
      c0 +
      (1 - alpha) *
      prep$eigvals

    adjustment <-
      (
        1 / shrunk -
        1 / c0
      ) *
      projected

    out <-
      out +
      drop(
        prep$V %*%
        adjustment
      )
  }

  out
}


# ============================================================
# 13.3 SHRINKAGE LDA PREDICTION
# ============================================================

predict_shrinkage_lda <- function(
    prep,
    X,
    alpha
) {

  X <- as.matrix(X)

  if (ncol(X) != prep$p) {
    stop(
      "X has an incompatible number of predictors.",
      call. = FALSE
    )
  }

  mean_difference <-
    prep$mu1 -
    prep$mu0

  coefficient <-
    solve_shrinkage_covariance(
      prep = prep,
      v = mean_difference,
      alpha = alpha
    )

  intercept <-
    -0.5 *
    sum(
      (
        prep$mu1 +
        prep$mu0
      ) *
      coefficient
    ) +
    log(
      prep$prior1 /
      prep$prior0
    )

  score <-
    drop(
      X %*%
      coefficient +
      intercept
    )

  prob <-
    plogis(score)

  list(
    pred =
      as.integer(
        prob >= 0.5
      ),
    prob = prob,
    score = score,
    condition_number =
      shrinkage_condition_number(
        eigvals = prep$eigvals,
        p = prep$p,
        alpha = alpha,
        tau_cov = prep$tau_cov
      )
  )
}


# ============================================================
# 13.4 TRAINING-ONLY CV FOR SHRINKAGE LDA
#
# Selection criterion:
#   pooled out-of-fold binary cross-entropy.
# ============================================================

tune_shrinkage_lda_cv <- function(
    X_train,
    y_train,
    alpha_grid =
      final_simulation_spec$lda_alpha_grid,
    foldid
) {

  X_train <- as.matrix(X_train)
  y_train <- as.integer(y_train)
  foldid <- as.integer(foldid)

  alpha_grid <-
    sort(
      unique(
        as.numeric(
          alpha_grid
        )
      )
    )

  if (
    length(y_train) != nrow(X_train) ||
    length(foldid) != nrow(X_train)
  ) {
    stop(
      "X_train, y_train and foldid have incompatible lengths.",
      call. = FALSE
    )
  }

  folds <-
    sort(
      unique(
        foldid
      )
    )

  oof_prob <-
    matrix(
      NA_real_,
      nrow = nrow(X_train),
      ncol = length(alpha_grid)
    )

  for (fold in folds) {

    val_idx <-
      which(
        foldid == fold
      )

    fit_idx <-
      which(
        foldid != fold
      )

    prep <-
      prepare_shrinkage_lda(
        X_train =
          X_train[
            fit_idx,
            ,
            drop = FALSE
          ],
        y_train =
          y_train[
            fit_idx
          ]
      )

    for (
      j in seq_along(
        alpha_grid
      )
    ) {

      fit <-
        predict_shrinkage_lda(
          prep = prep,
          X =
            X_train[
              val_idx,
              ,
              drop = FALSE
            ],
          alpha =
            alpha_grid[j]
        )

      oof_prob[
        val_idx,
        j
      ] <-
        fit$prob
    }
  }

  if (any(!is.finite(oof_prob))) {
    stop(
      "Incomplete shrinkage-LDA out-of-fold predictions.",
      call. = FALSE
    )
  }

  tuning_table <-
    tibble::tibble(
      alpha =
        alpha_grid,
      cv_validation_loss =
        vapply(
          seq_along(alpha_grid),
          function(j) {
            binary_log_loss(
              y_train,
              oof_prob[, j]
            )
          },
          numeric(1)
        ),
      cv_auc =
        vapply(
          seq_along(alpha_grid),
          function(j) {
            safe_auc(
              y_train,
              oof_prob[, j]
            )
          },
          numeric(1)
        ),
      cv_brier =
        vapply(
          seq_along(alpha_grid),
          function(j) {
            mean(
              (
                oof_prob[, j] -
                y_train
              )^2
            )
          },
          numeric(1)
        )
    ) |>
    dplyr::arrange(
      cv_validation_loss,
      alpha
    )

  selected <-
    tuning_table |>
    dplyr::slice_head(
      n = 1L
    )

  list(
    selected = selected,
    tuning_table = tuning_table,
    foldid = foldid,
    oof_prob = oof_prob
  )
}


# ============================================================
# 13.5 TRAINING-ONLY CV FOR RIDGE / ELASTIC-NET LOGISTIC
#
# glmnet parameterisation:
#
# lambda * {
#   mixing * ||beta||_1
#   + (1-mixing)/2 * ||beta||_2^2
# }.
#
# mixing = 0:
#   ridge.
#
# mixing in {0.25,0.50,0.75}:
#   genuine elastic-net candidates.
#
# The term "mixing" is used deliberately to avoid confusion
# with the covariance-shrinkage parameter alpha.
# ============================================================

tune_glmnet_logistic_cv <- function(
    X_train,
    y_train,
    mixing_grid,
    foldid,
    standardize = TRUE
) {

  X_train <- as.matrix(X_train)
  y_train <- as.integer(y_train)
  foldid <- as.integer(foldid)

  mixing_grid <-
    sort(
      unique(
        as.numeric(
          mixing_grid
        )
      )
    )

  if (
    length(y_train) != nrow(X_train) ||
    length(foldid) != nrow(X_train)
  ) {
    stop(
      "X_train, y_train and foldid have incompatible lengths.",
      call. = FALSE
    )
  }

  candidate_list <-
    vector(
      "list",
      length(mixing_grid)
    )

  path_list <-
    vector(
      "list",
      length(mixing_grid)
    )

  for (
    j in seq_along(
      mixing_grid
    )
  ) {

    mixing <-
      mixing_grid[j]

    cvfit <-
      glmnet::cv.glmnet(
        x = X_train,
        y = y_train,
        family = "binomial",
        alpha = mixing,
        foldid = foldid,
        nfolds =
          length(
            unique(
              foldid
            )
          ),
        type.measure = "deviance",
        standardize = standardize,
        intercept = TRUE,
        parallel = FALSE
      )

    idx <-
      which.min(
        cvfit$cvm
      )

    candidate_list[[j]] <-
      tibble::tibble(
        mixing = mixing,
        lambda =
          cvfit$lambda[idx],
        cv_deviance =
          cvfit$cvm[idx],
        cv_se =
          cvfit$cvsd[idx],
        lambda_boundary =
          idx %in%
          c(
            1L,
            length(
              cvfit$lambda
            )
          )
      )

    path_list[[j]] <-
      tibble::tibble(
        mixing = mixing,
        lambda =
          cvfit$lambda,
        cv_deviance =
          cvfit$cvm,
        cv_se =
          cvfit$cvsd
      )
  }

  candidate_table <-
    dplyr::bind_rows(
      candidate_list
    ) |>
    dplyr::arrange(
      cv_deviance,
      mixing
    )

  selected <-
    candidate_table |>
    dplyr::slice_head(
      n = 1L
    )

  list(
    selected = selected,
    candidate_table = candidate_table,
    tuning_path =
      dplyr::bind_rows(
        path_list
      ),
    foldid = foldid
  )
}


# ============================================================
# 13.6 FINAL RIDGE / ELASTIC-NET LOGISTIC FIT
# ============================================================

fit_predict_glmnet_logistic <- function(
    X_train,
    y_train,
    X_test,
    mixing,
    lambda,
    standardize = TRUE
) {

  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  y_train <- as.integer(y_train)

  fit <-
    glmnet::glmnet(
      x = X_train,
      y = y_train,
      family = "binomial",
      alpha = as.numeric(mixing),
      lambda = as.numeric(lambda),
      standardize = standardize,
      intercept = TRUE
    )

  prob <-
    as.numeric(
      predict(
        fit,
        newx = X_test,
        s = as.numeric(lambda),
        type = "response"
      )
    )

  list(
    pred =
      as.integer(
        prob >= 0.5
      ),
    prob = prob,
    model = fit
  )
}


# ============================================================
# 14. PRODUCTION MONTE CARLO RUNNERS
# ============================================================


# ============================================================
# 14.1 ONE NEURAL MONTE CARLO CELL
#
# One cell is:
#
#   scenario x p/n x network depth.
#
# Only ONE CR-DNN CV surface is fitted per replication.
#
# The ordinary ANN/DNN tuning result is extracted from
# alpha = 1, because the scale-preserving transformation makes
# this endpoint exactly equal to the ordinary dimension-
# normalised neural representation.
# ============================================================

run_neural_mc_cell <- function(
    scenario,
    n =
      spec$n,
    p_over_n = 1,
    depth = 3L,
    B =
      spec$B,
    rep_ids = NULL,
    n_test =
      spec$n_test,
    master_seed =
      spec$master_seed,
    spec = final_simulation_spec,
    verbose = TRUE
) {

  n <- as.integer(n)
  p <- as.integer(round(n * p_over_n))
  depth <- as.integer(depth)
  B <- as.integer(B)

  if (is.null(rep_ids)) {
    rep_ids <- seq_len(B)
  }

  rep_ids <- as.integer(rep_ids)

  if (
    length(rep_ids) < 1L ||
    any(!is.finite(rep_ids)) ||
    any(rep_ids < 1L | rep_ids > B) ||
    anyDuplicated(rep_ids)
  ) {
    stop(
      "rep_ids must be unique integers in 1:B.",
      call. = FALSE
    )
  }

  B_run <- length(rep_ids)

  signal_multiplier <-
    get_final_signal_multiplier(
      scenario
    )

  structural_seed <-
    get_structural_seed(
      p = p,
      master_seed = master_seed
    )

  dgp <-
    make_dgp(
      p = p,
      scenario = scenario,
      signal_multiplier =
        signal_multiplier,
      seed = structural_seed
    )

  raw_list <-
    vector(
      "list",
      2L * B_run
    )

  selection_list <-
    vector(
      "list",
      B_run
    )

  tuning_list <-
    vector(
      "list",
      B_run
    )

  row_id <- 1L

  for (
    ii in seq_along(rep_ids)
  ) {

    b <- rep_ids[ii]

    rep_seed <-
      get_replication_seed(
        p = p,
        rep = b,
        master_seed = master_seed
      )

    if (verbose) {
      message(
        sprintf(
          paste0(
            "NEURAL | scenario=%s | n=%d | p=%d | ",
            "p/n=%.2f | L=%d | rep=%d | task-rep=%d/%d"
          ),
          scenario,
          n,
          p,
          p / n,
          depth,
          b,
          ii,
          B_run
        )
      )
    }

    # --------------------------------------------------------
    # Common training / independent test samples
    # --------------------------------------------------------

    train_data <-
      sample_from_dgp(
        n = n,
        dgp = dgp,
        seed = rep_seed + 1L
      )

    test_data <-
      sample_from_dgp(
        n = n_test,
        dgp = dgp,
        seed = rep_seed + 2L
      )

    X_train_b <- train_data$X
    y_train_b <-
      as.integer(
        train_data$y
      )

    X_test_b <- test_data$X
    y_test_b <-
      as.integer(
        test_data$y
      )

    true_prob_test_b <-
      test_data$true_prob

    class_counts <-
      table(
        y_train_b
      )

    if (
      length(class_counts) != 2L ||
      min(class_counts) <
        spec$outer_folds
    ) {
      stop(
        "Insufficient class representation in replication ",
        b,
        ".",
        call. = FALSE
      )
    }

    # --------------------------------------------------------
    # Common training-only outer CV folds
    # --------------------------------------------------------

    cv_foldid_b <-
      make_stratified_foldid(
        y = y_train_b,
        k =
          spec$outer_folds,
        seed = rep_seed + 100L
      )

    # --------------------------------------------------------
    # Joint CR-DNN tuning surface
    # --------------------------------------------------------

    cr_cv_b <-
      tune_crdnn_cv(
        X_train = X_train_b,
        y_train = y_train_b,
        depth = depth,
        width =
          spec$hidden_width,
        alpha_grid =
          spec$alpha_grid,
        lambda_nn_grid =
          spec$lambda_nn_grid,
        foldid = cv_foldid_b,
        inner_validation_fraction =
          spec$
          inner_validation_fraction,
        penalty_tau =
          spec$penalty_tau,
        learning_rate =
          spec$learning_rate,
        epochs =
          spec$max_epochs,
        batch_size =
          spec$batch_size,
        patience =
          spec$patience,
        seed = rep_seed + 200L
      )

    # Full CR-DNN optimum
    cr_selected_b <-
      cr_cv_b$tuning_table |>
      dplyr::slice_head(
        n = 1L
      )

    # Ordinary ANN/DNN endpoint:
    # alpha = 1 exactly.
    dnn_selected_b <-
      cr_cv_b$tuning_table |>
      dplyr::filter(
        alpha == 1
      ) |>
      dplyr::arrange(
        cv_validation_loss,
        lambda_nn
      ) |>
      dplyr::slice_head(
        n = 1L
      )

    if (
      nrow(dnn_selected_b) != 1L ||
      nrow(cr_selected_b) != 1L
    ) {
      stop(
        "Could not select neural tuning parameters.",
        call. = FALSE
      )
    }

    # --------------------------------------------------------
    # Save the COMPLETE tuning surface
    # --------------------------------------------------------

    tuning_list[[ii]] <-
      cr_cv_b$tuning_table |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        depth = depth,
        rep = b,
        is_dnn_endpoint = alpha == 1,
        selected_cr =
          alpha ==
          cr_selected_b$alpha &
          lambda_nn ==
          cr_selected_b$lambda_nn,
        selected_dnn =
          alpha == 1 &
          lambda_nn ==
          dnn_selected_b$lambda_nn,
        alpha_boundary =
          alpha %in%
          range(
            spec$
            alpha_grid
          ),
        lambda_boundary =
          lambda_nn %in%
          range(
            spec$
            lambda_nn_grid
          )
      ) |>
      dplyr::select(
        scenario,
        n,
        p,
        p_over_n,
        depth,
        rep,
        dplyr::everything()
      )

    # --------------------------------------------------------
    # Common final training-only early-stopping split
    # --------------------------------------------------------

    final_split_b <-
      make_stratified_calibration_split(
        y = y_train_b,
        validation_fraction =
          spec$
          final_validation_fraction,
        seed = rep_seed + 300L
      )

    fit_idx_b <-
      final_split_b$fitting_index

    val_idx_b <-
      final_split_b$validation_index

    X_fit_b <-
      X_train_b[
        fit_idx_b,
        ,
        drop = FALSE
      ]

    y_fit_b <-
      y_train_b[
        fit_idx_b
      ]

    X_val_b <-
      X_train_b[
        val_idx_b,
        ,
        drop = FALSE
      ]

    y_val_b <-
      y_train_b[
        val_idx_b
      ]

    # --------------------------------------------------------
    # Ordinary ANN/DNN representation
    # --------------------------------------------------------

    dnn_val_prep_b <-
      centre_by_train(
        X_train = X_fit_b,
        X_test = X_val_b,
        dimension_normalise = TRUE
      )

    dnn_test_prep_b <-
      centre_by_train(
        X_train = X_fit_b,
        X_test = X_test_b,
        dimension_normalise = TRUE
      )

    # --------------------------------------------------------
    # CR-DNN scale-preserving representation
    # --------------------------------------------------------

    shrink_prep_b <-
      prepare_shrinkage(
        X_fit_b
      )

    cr_fit_repr_b <-
      apply_shrinkage(
        prep = shrink_prep_b,
        X = X_fit_b,
        alpha =
          cr_selected_b$alpha
      )

    cr_val_repr_b <-
      apply_shrinkage(
        prep = shrink_prep_b,
        X = X_val_b,
        alpha =
          cr_selected_b$alpha
      )

    cr_test_repr_b <-
      apply_shrinkage(
        prep = shrink_prep_b,
        X = X_test_b,
        alpha =
          cr_selected_b$alpha
      )

    model_seed <-
      rep_seed + 400L

    # --------------------------------------------------------
    # Matched final ANN/DNN fit
    # --------------------------------------------------------

    dnn_fit_b <-
      fit_predict_nn_final(
        X_fit =
          dnn_val_prep_b$X_train,
        y_fit = y_fit_b,
        X_validation =
          dnn_val_prep_b$X_test,
        y_validation = y_val_b,
        X_test =
          dnn_test_prep_b$X_test,
        depth = depth,
        width =
          spec$
          hidden_width,
        lambda_nn =
          dnn_selected_b$lambda_nn,
        penalty_tau =
          spec$
          penalty_tau,
        learning_rate =
          spec$
          learning_rate,
        epochs =
          spec$
          max_epochs,
        batch_size =
          spec$
          batch_size,
        patience =
          spec$
          patience,
        seed = model_seed,
        verbose = FALSE
      )

    # --------------------------------------------------------
    # Matched final CR-DNN fit
    # --------------------------------------------------------

    endpoint_probability_difference <-
      NA_real_

    at_dnn_endpoint <-
      isTRUE(
        all.equal(
          cr_selected_b$alpha,
          1
        )
      )

    if (at_dnn_endpoint) {

      # At alpha = 1 the CR-DNN representation, objective, architecture,
      # tuning parameter and training sample are exactly those of the
      # ordinary DNN.  A second stochastic TensorFlow fit is therefore
      # neither statistically distinct nor numerically useful.  Reuse the
      # matched DNN fit so that the theoretical endpoint identity is
      # enforced by construction rather than tested across two potentially
      # non-bitwise-deterministic optimisation runs.
      if (
        !isTRUE(
          all.equal(
            cr_selected_b$lambda_nn,
            dnn_selected_b$lambda_nn
          )
        )
      ) {
        stop(
          "At alpha=1 the CR-DNN and DNN selected lambdas must coincide.",
          call. = FALSE
        )
      }

      cr_fit_b <- dnn_fit_b
      endpoint_probability_difference <- 0

    } else {

      cr_fit_b <-
        fit_predict_nn_final(
          X_fit =
            cr_fit_repr_b$Z,
          y_fit = y_fit_b,
          X_validation =
            cr_val_repr_b$Z,
          y_validation = y_val_b,
          X_test =
            cr_test_repr_b$Z,
          depth = depth,
          width =
            spec$
            hidden_width,
          lambda_nn =
            cr_selected_b$lambda_nn,
          penalty_tau =
            spec$
            penalty_tau,
          learning_rate =
            spec$
            learning_rate,
          epochs =
            spec$
            max_epochs,
          batch_size =
            spec$
            batch_size,
          patience =
            spec$
            patience,
          seed = model_seed,
          verbose = FALSE
        )
    }

    # --------------------------------------------------------
    # Exact endpoint invariant
    # --------------------------------------------------------

    if (at_dnn_endpoint) {
      if (
        !is.finite(endpoint_probability_difference) ||
        endpoint_probability_difference != 0
      ) {
        stop(
          "alpha=1 endpoint equivalence failed.",
          call. = FALSE
        )
      }
    }

    # --------------------------------------------------------
    # Common independent test metrics
    # --------------------------------------------------------

    dnn_met_b <-
      evaluate_predictions(
        y_true = y_test_b,
        pred = dnn_fit_b$pred,
        prob = dnn_fit_b$prob,
        true_prob =
          true_prob_test_b
      )

    cr_met_b <-
      evaluate_predictions(
        y_true = y_test_b,
        pred = cr_fit_b$pred,
        prob = cr_fit_b$prob,
        true_prob =
          true_prob_test_b
      )

    method_label <-
      network_method_label(
        depth
      )

    raw_list[[row_id]] <-
      dnn_met_b |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        depth = depth,
        rep = b,
        method = "DNN",
        method_label = method_label,
        method_group = "neural",
        selected_alpha = 1,
        selected_lambda =
          dnn_selected_b$lambda_nn,
        selected_mixing = NA_real_,
        best_epoch =
          dnn_fit_b$best_epoch,
        epochs_used =
          dnn_fit_b$epochs_used,
        probability_sd =
          sd(
            dnn_fit_b$prob
          ),
        condition_number =
          NA_real_,
        endpoint_probability_difference =
          endpoint_probability_difference
      )

    row_id <- row_id + 1L

    raw_list[[row_id]] <-
      cr_met_b |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        depth = depth,
        rep = b,
        method = "CR-DNN",
        method_label =
          paste0(
            "CR-DNN (L=",
            depth,
            ")"
          ),
        method_group = "neural",
        selected_alpha =
          cr_selected_b$alpha,
        selected_lambda =
          cr_selected_b$lambda_nn,
        selected_mixing = NA_real_,
        best_epoch =
          cr_fit_b$best_epoch,
        epochs_used =
          cr_fit_b$epochs_used,
        probability_sd =
          sd(
            cr_fit_b$prob
          ),
        condition_number =
          cr_fit_repr_b$
          condition_number,
        endpoint_probability_difference =
          endpoint_probability_difference
      )

    row_id <- row_id + 1L

    # --------------------------------------------------------
    # Selection summary
    # --------------------------------------------------------

    selection_list[[ii]] <-
      tibble::tibble(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        depth = depth,
        rep = b,
        dnn_lambda =
          dnn_selected_b$lambda_nn,
        dnn_cv_bce =
          dnn_selected_b$
          cv_validation_loss,
        cr_alpha =
          cr_selected_b$alpha,
        cr_lambda =
          cr_selected_b$lambda_nn,
        cr_cv_bce =
          cr_selected_b$
          cv_validation_loss,
        cr_alpha_interior =
          cr_selected_b$alpha < 1,
        cr_alpha_identity =
          cr_selected_b$alpha == 1,
        cr_alpha_lower_boundary =
          cr_selected_b$alpha ==
          min(
            spec$
            alpha_grid
          ),
        cr_lambda_boundary =
          cr_selected_b$lambda_nn %in%
          range(
            spec$
            lambda_nn_grid
          ),
        dnn_lambda_boundary =
          dnn_selected_b$lambda_nn %in%
          range(
            spec$
            lambda_nn_grid
          ),
        endpoint_probability_difference =
          endpoint_probability_difference
      )
  }

  raw_results <-
    dplyr::bind_rows(
      raw_list
    )

  selection_results <-
    dplyr::bind_rows(
      selection_list
    )

  tuning_results <-
    dplyr::bind_rows(
      tuning_list
    )

  # ----------------------------------------------------------
  # Bayes invariants within replication
  # ----------------------------------------------------------

  invariant_check <-
    raw_results |>
    dplyr::group_by(
      rep
    ) |>
    dplyr::summarise(
      methods =
        dplyr::n_distinct(
          method
        ),
      bayes_risks =
        dplyr::n_distinct(
          signif(
            bayes_class_risk,
            12
          )
        ),
      bayes_logistic_risks =
        dplyr::n_distinct(
          signif(
            bayes_logistic_risk,
            12
          )
        ),
      bayes_briers =
        dplyr::n_distinct(
          signif(
            bayes_brier,
            12
          )
        ),
      .groups = "drop"
    )

  if (
    any(
      invariant_check$methods != 2L |
      invariant_check$bayes_risks != 1L |
      invariant_check$bayes_logistic_risks != 1L |
      invariant_check$bayes_briers != 1L
    )
  ) {
    stop(
      "Neural common-test Bayes invariant failed.",
      call. = FALSE
    )
  }

  # ----------------------------------------------------------
  # Paired CR-DNN minus ordinary ANN/DNN contrasts
  # ----------------------------------------------------------

  dnn_rows <-
    raw_results |>
    dplyr::filter(
      method == "DNN"
    ) |>
    dplyr::select(
      scenario,
      n,
      p,
      p_over_n,
      depth,
      rep,
      dnn_risk = risk,
      dnn_conditional_risk =
        conditional_risk,
      dnn_excess_risk =
        excess_class_risk,
      dnn_auc = auc,
      dnn_brier = brier,
      dnn_logistic_risk =
        logistic_risk
    )

  cr_rows <-
    raw_results |>
    dplyr::filter(
      method == "CR-DNN"
    ) |>
    dplyr::select(
      scenario,
      n,
      p,
      p_over_n,
      depth,
      rep,
      cr_risk = risk,
      cr_conditional_risk =
        conditional_risk,
      cr_excess_risk =
        excess_class_risk,
      cr_auc = auc,
      cr_brier = brier,
      cr_logistic_risk =
        logistic_risk
    )

  contrasts <-
    dnn_rows |>
    dplyr::inner_join(
      cr_rows,
      by = c(
        "scenario",
        "n",
        "p",
        "p_over_n",
        "depth",
        "rep"
      )
    ) |>
    dplyr::transmute(
      scenario,
      n,
      p,
      p_over_n,
      depth,
      rep,
      contrast = "CR-DNN - DNN",
      risk_difference =
        cr_risk -
        dnn_risk,
      conditional_risk_difference =
        cr_conditional_risk -
        dnn_conditional_risk,
      excess_risk_difference =
        cr_excess_risk -
        dnn_excess_risk,
      auc_difference =
        cr_auc -
        dnn_auc,
      brier_difference =
        cr_brier -
        dnn_brier,
      logistic_risk_difference =
        cr_logistic_risk -
        dnn_logistic_risk
    )

  dgp_diagnostics <-
    tibble::tibble(
      scenario = scenario,
      n = n,
      p = p,
      p_over_n = p / n,
      depth = depth,
      signal_multiplier =
        dgp$signal_multiplier,
      effective_signal =
        dgp$effective_signal,
      response_regime =
        dgp$response_regime,
      covariance_regime =
        dgp$covariance_regime,
      delta_operator =
        dgp$delta_operator,
      delta_relative =
        dgp$delta_relative,
      population_condition_number =
        dgp$kappa_population,
      mean_population_variance =
        dgp$mean_population_variance
    )

  list(
    raw = raw_results,
    selection = selection_results,
    tuning = tuning_results,
    contrasts = contrasts,
    invariants = invariant_check,
    dgp = dgp_diagnostics
  )
}


# ============================================================
# 14.2 ONE DEPTH-INDEPENDENT LINEAR BENCHMARK CELL
#
# One cell is:
#
#   scenario x p/n.
#
# The same generated data and CV fold construction used by the
# neural runner are recovered from the common seed map.
# ============================================================

run_linear_benchmark_cell <- function(
    scenario,
    n =
      final_simulation_spec$n,
    p_over_n = 1,
    B =
      final_simulation_spec$B,
    n_test =
      final_simulation_spec$n_test,
    master_seed =
      final_simulation_spec$master_seed,
    verbose = TRUE
) {

  n <- as.integer(n)
  p <- as.integer(round(n * p_over_n))
  B <- as.integer(B)

  signal_multiplier <-
    get_final_signal_multiplier(
      scenario
    )

  structural_seed <-
    get_structural_seed(
      p = p,
      master_seed = master_seed
    )

  dgp <-
    make_dgp(
      p = p,
      scenario = scenario,
      signal_multiplier =
        signal_multiplier,
      seed = structural_seed
    )

  raw_list <-
    vector(
      "list",
      3L * B
    )

  selection_list <-
    vector(
      "list",
      3L * B
    )

  tuning_list <- list()
  tuning_id <- 1L
  row_id <- 1L
  selection_id <- 1L

  for (
    b in seq_len(B)
  ) {

    rep_seed <-
      get_replication_seed(
        p = p,
        rep = b,
        master_seed = master_seed
      )

    if (verbose) {
      message(
        sprintf(
          paste0(
            "LINEAR | scenario=%s | n=%d | p=%d | ",
            "p/n=%.2f | rep=%d/%d"
          ),
          scenario,
          n,
          p,
          p / n,
          b,
          B
        )
      )
    }

    train_data <-
      sample_from_dgp(
        n = n,
        dgp = dgp,
        seed = rep_seed + 1L
      )

    test_data <-
      sample_from_dgp(
        n = n_test,
        dgp = dgp,
        seed = rep_seed + 2L
      )

    X_train_b <- train_data$X
    y_train_b <-
      as.integer(
        train_data$y
      )

    X_test_b <- test_data$X
    y_test_b <-
      as.integer(
        test_data$y
      )

    true_prob_test_b <-
      test_data$true_prob

    class_counts <-
      table(
        y_train_b
      )

    if (
      length(class_counts) != 2L ||
      min(class_counts) <
        final_simulation_spec$outer_folds
    ) {
      stop(
        "Insufficient class representation in replication ",
        b,
        ".",
        call. = FALSE
      )
    }

    cv_foldid_b <-
      make_stratified_foldid(
        y = y_train_b,
        k =
          final_simulation_spec$outer_folds,
        seed = rep_seed + 100L
      )

    # ========================================================
    # Shrinkage LDA
    # ========================================================

    lda_cv_b <-
      tune_shrinkage_lda_cv(
        X_train = X_train_b,
        y_train = y_train_b,
        alpha_grid =
          final_simulation_spec$lda_alpha_grid,
        foldid = cv_foldid_b
      )

    lda_selected_b <-
      lda_cv_b$selected

    lda_prep_b <-
      prepare_shrinkage_lda(
        X_train_b,
        y_train_b
      )

    lda_fit_b <-
      predict_shrinkage_lda(
        prep = lda_prep_b,
        X = X_test_b,
        alpha =
          lda_selected_b$alpha
      )

    lda_met_b <-
      evaluate_predictions(
        y_true = y_test_b,
        pred = lda_fit_b$pred,
        prob = lda_fit_b$prob,
        true_prob =
          true_prob_test_b
      )

    raw_list[[row_id]] <-
      lda_met_b |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        depth = NA_integer_,
        rep = b,
        method = "ShrinkageLDA",
        method_label = "Shrinkage LDA",
        method_group = "linear",
        selected_alpha =
          lda_selected_b$alpha,
        selected_lambda = NA_real_,
        selected_mixing = NA_real_,
        best_epoch = NA_integer_,
        epochs_used = NA_integer_,
        probability_sd =
          sd(
            lda_fit_b$prob
          ),
        condition_number =
          lda_fit_b$
          condition_number
      )

    row_id <- row_id + 1L

    selection_list[[selection_id]] <-
      tibble::tibble(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        rep = b,
        method = "ShrinkageLDA",
        selected_alpha =
          lda_selected_b$alpha,
        selected_mixing = NA_real_,
        selected_lambda = NA_real_,
        cv_score =
          lda_selected_b$
          cv_validation_loss,
        alpha_boundary =
          lda_selected_b$alpha %in%
          range(
            final_simulation_spec$
            alpha_grid
          ),
        lambda_boundary = NA
      )

    selection_id <-
      selection_id + 1L

    tuning_list[[tuning_id]] <-
      lda_cv_b$tuning_table |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        rep = b,
        method = "ShrinkageLDA",
        mixing = NA_real_,
        lambda = NA_real_,
        selected =
          alpha ==
          lda_selected_b$alpha
      ) |>
      dplyr::select(
        scenario,
        n,
        p,
        p_over_n,
        rep,
        method,
        alpha,
        mixing,
        lambda,
        selected,
        dplyr::everything()
      )

    tuning_id <- tuning_id + 1L

    # ========================================================
    # Ridge logistic regression
    # ========================================================

    ridge_cv_b <-
      tune_glmnet_logistic_cv(
        X_train = X_train_b,
        y_train = y_train_b,
        mixing_grid = 0,
        foldid = cv_foldid_b,
        standardize = TRUE
      )

    ridge_selected_b <-
      ridge_cv_b$selected

    ridge_fit_b <-
      fit_predict_glmnet_logistic(
        X_train = X_train_b,
        y_train = y_train_b,
        X_test = X_test_b,
        mixing = 0,
        lambda =
          ridge_selected_b$lambda,
        standardize = TRUE
      )

    ridge_met_b <-
      evaluate_predictions(
        y_true = y_test_b,
        pred = ridge_fit_b$pred,
        prob = ridge_fit_b$prob,
        true_prob =
          true_prob_test_b
      )

    raw_list[[row_id]] <-
      ridge_met_b |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        depth = NA_integer_,
        rep = b,
        method = "RidgeLogit",
        method_label =
          "Ridge logistic",
        method_group = "linear",
        selected_alpha = NA_real_,
        selected_lambda =
          ridge_selected_b$lambda,
        selected_mixing = 0,
        best_epoch = NA_integer_,
        epochs_used = NA_integer_,
        probability_sd =
          sd(
            ridge_fit_b$prob
          ),
        condition_number = NA_real_
      )

    row_id <- row_id + 1L

    selection_list[[selection_id]] <-
      tibble::tibble(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        rep = b,
        method = "RidgeLogit",
        selected_alpha = NA_real_,
        selected_mixing = 0,
        selected_lambda =
          ridge_selected_b$lambda,
        cv_score =
          ridge_selected_b$
          cv_deviance,
        alpha_boundary = NA,
        lambda_boundary =
          ridge_selected_b$
          lambda_boundary
      )

    selection_id <-
      selection_id + 1L

    tuning_list[[tuning_id]] <-
      ridge_cv_b$tuning_path |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        rep = b,
        method = "RidgeLogit",
        alpha = NA_real_,
        selected =
          mixing == 0 &
          lambda ==
          ridge_selected_b$lambda
      ) |>
      dplyr::select(
        scenario,
        n,
        p,
        p_over_n,
        rep,
        method,
        alpha,
        mixing,
        lambda,
        selected,
        dplyr::everything()
      )

    tuning_id <- tuning_id + 1L

    # ========================================================
    # Elastic-net logistic regression
    # ========================================================

    enet_cv_b <-
      tune_glmnet_logistic_cv(
        X_train = X_train_b,
        y_train = y_train_b,
        mixing_grid =
          final_simulation_spec$
          enet_mixing_grid,
        foldid = cv_foldid_b,
        standardize = TRUE
      )

    enet_selected_b <-
      enet_cv_b$selected

    enet_fit_b <-
      fit_predict_glmnet_logistic(
        X_train = X_train_b,
        y_train = y_train_b,
        X_test = X_test_b,
        mixing =
          enet_selected_b$mixing,
        lambda =
          enet_selected_b$lambda,
        standardize = TRUE
      )

    enet_met_b <-
      evaluate_predictions(
        y_true = y_test_b,
        pred = enet_fit_b$pred,
        prob = enet_fit_b$prob,
        true_prob =
          true_prob_test_b
      )

    raw_list[[row_id]] <-
      enet_met_b |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        depth = NA_integer_,
        rep = b,
        method = "ElasticNetLogit",
        method_label =
          "Elastic-net logistic",
        method_group = "linear",
        selected_alpha = NA_real_,
        selected_lambda =
          enet_selected_b$lambda,
        selected_mixing =
          enet_selected_b$mixing,
        best_epoch = NA_integer_,
        epochs_used = NA_integer_,
        probability_sd =
          sd(
            enet_fit_b$prob
          ),
        condition_number = NA_real_
      )

    row_id <- row_id + 1L

    selection_list[[selection_id]] <-
      tibble::tibble(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        rep = b,
        method = "ElasticNetLogit",
        selected_alpha = NA_real_,
        selected_mixing =
          enet_selected_b$mixing,
        selected_lambda =
          enet_selected_b$lambda,
        cv_score =
          enet_selected_b$
          cv_deviance,
        alpha_boundary = NA,
        lambda_boundary =
          enet_selected_b$
          lambda_boundary
      )

    selection_id <-
      selection_id + 1L

    tuning_list[[tuning_id]] <-
      enet_cv_b$tuning_path |>
      dplyr::mutate(
        scenario = scenario,
        n = n,
        p = p,
        p_over_n = p / n,
        rep = b,
        method = "ElasticNetLogit",
        alpha = NA_real_,
        selected =
          mixing ==
          enet_selected_b$mixing &
          lambda ==
          enet_selected_b$lambda
      ) |>
      dplyr::select(
        scenario,
        n,
        p,
        p_over_n,
        rep,
        method,
        alpha,
        mixing,
        lambda,
        selected,
        dplyr::everything()
      )

    tuning_id <- tuning_id + 1L
  }

  raw_results <-
    dplyr::bind_rows(
      raw_list
    )

  selection_results <-
    dplyr::bind_rows(
      selection_list
    )

  tuning_results <-
    dplyr::bind_rows(
      tuning_list
    )

  invariant_check <-
    raw_results |>
    dplyr::group_by(
      rep
    ) |>
    dplyr::summarise(
      methods =
        dplyr::n_distinct(
          method
        ),
      bayes_risks =
        dplyr::n_distinct(
          signif(
            bayes_class_risk,
            12
          )
        ),
      bayes_logistic_risks =
        dplyr::n_distinct(
          signif(
            bayes_logistic_risk,
            12
          )
        ),
      bayes_briers =
        dplyr::n_distinct(
          signif(
            bayes_brier,
            12
          )
        ),
      .groups = "drop"
    )

  if (
    any(
      invariant_check$methods != 3L |
      invariant_check$bayes_risks != 1L |
      invariant_check$bayes_logistic_risks != 1L |
      invariant_check$bayes_briers != 1L
    )
  ) {
    stop(
      "Linear-benchmark common-test Bayes invariant failed.",
      call. = FALSE
    )
  }

  dgp_diagnostics <-
    tibble::tibble(
      scenario = scenario,
      n = n,
      p = p,
      p_over_n = p / n,
      signal_multiplier =
        dgp$signal_multiplier,
      effective_signal =
        dgp$effective_signal,
      response_regime =
        dgp$response_regime,
      covariance_regime =
        dgp$covariance_regime,
      delta_operator =
        dgp$delta_operator,
      delta_relative =
        dgp$delta_relative,
      population_condition_number =
        dgp$kappa_population,
      mean_population_variance =
        dgp$mean_population_variance
    )

  list(
    raw = raw_results,
    selection = selection_results,
    tuning = tuning_results,
    invariants = invariant_check,
    dgp = dgp_diagnostics
  )
}


# ============================================================
# 14.3 PRODUCTION DESIGN TABLES
# ============================================================

make_production_design <- function(
    spec =
      final_simulation_spec
) {

  neural <-
    tidyr::expand_grid(
      scenario =
        spec$scenarios,
      p_over_n =
        spec$p_over_n,
      depth =
        spec$depth
    ) |>
    dplyr::mutate(
      cell_id =
        dplyr::row_number(),
      n =
        as.integer(
          spec$n
        ),
      p =
        as.integer(
          round(
            n *
            p_over_n
          )
        ),
      B =
        as.integer(
          spec$B
        ),
      n_test =
        as.integer(
          spec$n_test
        )
    ) |>
    dplyr::select(
      cell_id,
      dplyr::everything()
    )

  linear <-
    tidyr::expand_grid(
      scenario =
        spec$scenarios,
      p_over_n =
        spec$p_over_n
    ) |>
    dplyr::mutate(
      cell_id =
        dplyr::row_number(),
      n =
        as.integer(
          spec$n
        ),
      p =
        as.integer(
          round(
            n *
            p_over_n
          )
        ),
      B =
        as.integer(
          spec$B
        ),
      n_test =
        as.integer(
          spec$n_test
        )
    ) |>
    dplyr::select(
      cell_id,
      dplyr::everything()
    )

  list(
    neural = neural,
    linear = linear
  )
}


# ============================================================
# 14.4 CELL-LEVEL FILE NAMES
# ============================================================

make_cell_stub <- function(
    layer,
    scenario,
    p_over_n,
    depth = NULL
) {

  ratio_text <-
    gsub(
      "\\.",
      "p",
      format(
        p_over_n,
        trim = TRUE,
        scientific = FALSE
      )
    )

  if (
    identical(
      layer,
      "neural"
    )
  ) {

    paste0(
      "neural__",
      scenario,
      "__pn_",
      ratio_text,
      "__L_",
      as.integer(depth)
    )

  } else {

    paste0(
      "linear__",
      scenario,
      "__pn_",
      ratio_text
    )
  }
}


# ============================================================
# 14.5 SAVE ONE COMPLETED CELL
# ============================================================

save_simulation_cell <- function(
    result,
    layer,
    scenario,
    p_over_n,
    depth = NULL,
    results_dir = "results"
) {

  dir.create(
    results_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  stub <-
    make_cell_stub(
      layer = layer,
      scenario = scenario,
      p_over_n = p_over_n,
      depth = depth
    )

  write.csv(
    result$raw,
    file.path(
      results_dir,
      paste0(
        stub,
        "__raw.csv"
      )
    ),
    row.names = FALSE
  )

  write.csv(
    result$selection,
    file.path(
      results_dir,
      paste0(
        stub,
        "__selection.csv"
      )
    ),
    row.names = FALSE
  )

  write.csv(
    result$tuning,
    file.path(
      results_dir,
      paste0(
        stub,
        "__tuning.csv"
      )
    ),
    row.names = FALSE
  )

  write.csv(
    result$invariants,
    file.path(
      results_dir,
      paste0(
        stub,
        "__invariants.csv"
      )
    ),
    row.names = FALSE
  )

  write.csv(
    result$dgp,
    file.path(
      results_dir,
      paste0(
        stub,
        "__dgp.csv"
      )
    ),
    row.names = FALSE
  )

  if (
    !is.null(
      result$contrasts
    )
  ) {
    write.csv(
      result$contrasts,
      file.path(
        results_dir,
        paste0(
          stub,
          "__contrasts.csv"
        )
      ),
      row.names = FALSE
    )
  }

  invisible(stub)
}


# ============================================================
# 14.6 RUN ONE NEURAL CELL FROM THE DESIGN TABLE
#
# Intended HPC use:
#
#   run_neural_production_cell(cell_id = ...)
# ============================================================

run_neural_production_cell <- function(
    cell_id,
    results_dir = "results",
    overwrite = FALSE,
    spec =
      final_simulation_spec,
    verbose = TRUE
) {

  design <-
    make_production_design(
      spec
    )$neural

  row <-
    design[
      design$cell_id ==
      as.integer(cell_id),
      ,
      drop = FALSE
    ]

  if (nrow(row) != 1L) {
    stop(
      "cell_id must identify exactly one neural design cell.",
      call. = FALSE
    )
  }

  stub <-
    make_cell_stub(
      layer = "neural",
      scenario =
        row$scenario,
      p_over_n =
        row$p_over_n,
      depth =
        row$depth
    )

  raw_file <-
    file.path(
      results_dir,
      paste0(
        stub,
        "__raw.csv"
      )
    )

  if (
    file.exists(raw_file) &&
    !overwrite
  ) {
    message(
      "Skipping completed neural cell: ",
      stub
    )
    return(
      invisible(NULL)
    )
  }

  result <-
    run_neural_mc_cell(
      scenario =
        row$scenario,
      n =
        row$n,
      p_over_n =
        row$p_over_n,
      depth =
        row$depth,
      B =
        row$B,
      n_test =
        row$n_test,
      master_seed =
        spec$master_seed,
      spec = spec,
      verbose =
        verbose
    )

  save_simulation_cell(
    result = result,
    layer = "neural",
    scenario =
      row$scenario,
    p_over_n =
      row$p_over_n,
    depth =
      row$depth,
    results_dir =
      results_dir
  )

  invisible(result)
}


# ============================================================
# 14.7 RUN ONE LINEAR BENCHMARK CELL FROM THE DESIGN TABLE
# ============================================================

run_linear_production_cell <- function(
    cell_id,
    results_dir = "results",
    overwrite = FALSE,
    spec =
      final_simulation_spec,
    verbose = TRUE
) {

  design <-
    make_production_design(
      spec
    )$linear

  row <-
    design[
      design$cell_id ==
      as.integer(cell_id),
      ,
      drop = FALSE
    ]

  if (nrow(row) != 1L) {
    stop(
      "cell_id must identify exactly one linear design cell.",
      call. = FALSE
    )
  }

  stub <-
    make_cell_stub(
      layer = "linear",
      scenario =
        row$scenario,
      p_over_n =
        row$p_over_n
    )

  raw_file <-
    file.path(
      results_dir,
      paste0(
        stub,
        "__raw.csv"
      )
    )

  if (
    file.exists(raw_file) &&
    !overwrite
  ) {
    message(
      "Skipping completed linear cell: ",
      stub
    )
    return(
      invisible(NULL)
    )
  }

  result <-
    run_linear_benchmark_cell(
      scenario =
        row$scenario,
      n =
        row$n,
      p_over_n =
        row$p_over_n,
      B =
        row$B,
      n_test =
        row$n_test,
      master_seed =
        spec$master_seed,
      verbose =
        verbose
    )

  save_simulation_cell(
    result = result,
    layer = "linear",
    scenario =
      row$scenario,
    p_over_n =
      row$p_over_n,
    results_dir =
      results_dir
  )

  invisible(result)
}


# ============================================================
# 15. COMPILE AND SUMMARISE PRODUCTION RESULTS
# ============================================================


# ============================================================
# 15.1 READ A RESULT-FILE FAMILY
# ============================================================

read_result_family <- function(
    results_dir,
    suffix
) {

  files <-
    list.files(
      results_dir,
      pattern =
        paste0(
          suffix,
          "\\.csv$"
        ),
      full.names = TRUE
    )

  if (length(files) == 0L) {
    return(
      tibble::tibble()
    )
  }

  dplyr::bind_rows(
    lapply(
      files,
      read.csv,
      stringsAsFactors = FALSE
    )
  )
}


# ============================================================
# 15.2 COMPILE ALL CELL FILES
# ============================================================

compile_production_results <- function(
    results_dir = "results",
    compiled_dir =
      file.path(
        results_dir,
        "compiled"
      ),
    write_output = TRUE
) {

  raw <-
    read_result_family(
      results_dir,
      "__raw"
    )

  selection <-
    read_result_family(
      results_dir,
      "__selection"
    )

  tuning <-
    read_result_family(
      results_dir,
      "__tuning"
    )

  invariants <-
    read_result_family(
      results_dir,
      "__invariants"
    )

  dgp <-
    read_result_family(
      results_dir,
      "__dgp"
    )

  contrasts <-
    read_result_family(
      results_dir,
      "__contrasts"
    )

  if (
    nrow(raw) == 0L
  ) {
    stop(
      "No production raw-result files were found.",
      call. = FALSE
    )
  }

  compiled <-
    list(
      raw = raw,
      selection = selection,
      tuning = tuning,
      invariants = invariants,
      dgp = dgp,
      neural_contrasts = contrasts
    )

  if (write_output) {

    dir.create(
      compiled_dir,
      recursive = TRUE,
      showWarnings = FALSE
    )

    write.csv(
      raw,
      file.path(
        compiled_dir,
        "production_raw.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      selection,
      file.path(
        compiled_dir,
        "production_selection.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      tuning,
      file.path(
        compiled_dir,
        "production_tuning.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      invariants,
      file.path(
        compiled_dir,
        "production_invariants.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      dgp,
      file.path(
        compiled_dir,
        "production_dgp.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      contrasts,
      file.path(
        compiled_dir,
        "production_neural_contrasts.csv"
      ),
      row.names = FALSE
    )
  }

  compiled
}


# ============================================================
# 15.3 METHOD-LEVEL MONTE CARLO PERFORMANCE SUMMARY
# ============================================================

summarise_method_performance <- function(
    raw
) {

  raw |>
    dplyr::group_by(
      scenario,
      n,
      p,
      p_over_n,
      depth,
      method,
      method_label,
      method_group
    ) |>
    dplyr::summarise(
      B =
        dplyr::n(),
      mean_risk =
        mean(
          risk,
          na.rm = TRUE
        ),
      mcse_risk =
        stats::sd(
          risk,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_conditional_risk =
        mean(
          conditional_risk,
          na.rm = TRUE
        ),
      mcse_conditional_risk =
        stats::sd(
          conditional_risk,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_bayes_class_risk =
        mean(
          bayes_class_risk,
          na.rm = TRUE
        ),
      mean_excess_class_risk =
        mean(
          excess_class_risk,
          na.rm = TRUE
        ),
      mean_auc =
        mean(
          auc,
          na.rm = TRUE
        ),
      mcse_auc =
        stats::sd(
          auc,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_brier =
        mean(
          brier,
          na.rm = TRUE
        ),
      mcse_brier =
        stats::sd(
          brier,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_logistic_risk =
        mean(
          logistic_risk,
          na.rm = TRUE
        ),
      mcse_logistic_risk =
        stats::sd(
          logistic_risk,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_surrogate_excess_risk =
        mean(
          surrogate_excess_risk,
          na.rm = TRUE
        ),
      mean_probability_sd =
        mean(
          probability_sd,
          na.rm = TRUE
        ),
      mean_condition_number =
        if (
          all(
            is.na(
              condition_number
            )
          )
        ) {
          NA_real_
        } else {
          mean(
            condition_number,
            na.rm = TRUE
          )
        },
      .groups = "drop"
    )
}


# ============================================================
# 15.4 PRIMARY PAIRED CR-DNN MINUS ANN/DNN SUMMARY
#
# Signs:
#   risk/Brier/logistic < 0 favours CR-DNN;
#   AUC > 0 favours CR-DNN.
# ============================================================

summarise_neural_contrasts <- function(
    neural_contrasts
) {

  neural_contrasts |>
    dplyr::group_by(
      scenario,
      n,
      p,
      p_over_n,
      depth,
      contrast
    ) |>
    dplyr::summarise(
      B =
        dplyr::n(),
      mean_risk_difference =
        mean(
          risk_difference,
          na.rm = TRUE
        ),
      mcse_risk_difference =
        stats::sd(
          risk_difference,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_conditional_risk_difference =
        mean(
          conditional_risk_difference,
          na.rm = TRUE
        ),
      mcse_conditional_risk_difference =
        stats::sd(
          conditional_risk_difference,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_auc_difference =
        mean(
          auc_difference,
          na.rm = TRUE
        ),
      mcse_auc_difference =
        stats::sd(
          auc_difference,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_brier_difference =
        mean(
          brier_difference,
          na.rm = TRUE
        ),
      mcse_brier_difference =
        stats::sd(
          brier_difference,
          na.rm = TRUE
        ) /
        sqrt(B),
      mean_logistic_risk_difference =
        mean(
          logistic_risk_difference,
          na.rm = TRUE
        ),
      mcse_logistic_risk_difference =
        stats::sd(
          logistic_risk_difference,
          na.rm = TRUE
        ) /
        sqrt(B),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      risk_ci_low =
        mean_risk_difference -
        1.96 *
        mcse_risk_difference,
      risk_ci_high =
        mean_risk_difference +
        1.96 *
        mcse_risk_difference,
      auc_ci_low =
        mean_auc_difference -
        1.96 *
        mcse_auc_difference,
      auc_ci_high =
        mean_auc_difference +
        1.96 *
        mcse_auc_difference,
      brier_ci_low =
        mean_brier_difference -
        1.96 *
        mcse_brier_difference,
      brier_ci_high =
        mean_brier_difference +
        1.96 *
        mcse_brier_difference
    )
}


# ============================================================
# 15.5 CR-DNN SHRINKAGE-SELECTION SUMMARY
# ============================================================

summarise_crdnn_selection <- function(
    selection
) {

  selection |>
    dplyr::filter(
      !is.na(
        cr_alpha
      )
    ) |>
    dplyr::group_by(
      scenario,
      n,
      p,
      p_over_n,
      depth
    ) |>
    dplyr::summarise(
      B =
        dplyr::n(),
      interior_alpha_frequency =
        mean(
          cr_alpha < 1,
          na.rm = TRUE
        ),
      identity_endpoint_frequency =
        mean(
          cr_alpha == 1,
          na.rm = TRUE
        ),
      lower_boundary_frequency =
        mean(
          cr_alpha ==
          min(
            final_simulation_spec$
            alpha_grid
          ),
          na.rm = TRUE
        ),
      mean_selected_alpha =
        mean(
          cr_alpha,
          na.rm = TRUE
        ),
      median_selected_alpha =
        stats::median(
          cr_alpha,
          na.rm = TRUE
        ),
      lambda_boundary_frequency =
        mean(
          cr_lambda_boundary,
          na.rm = TRUE
        ),
      .groups = "drop"
    )
}


# ============================================================
# 15.6 LINEAR-BENCHMARK TUNING SUMMARY
# ============================================================

summarise_linear_selection <- function(
    selection
) {

  selection |>
    dplyr::filter(
      method %in%
      c(
        "ShrinkageLDA",
        "RidgeLogit",
        "ElasticNetLogit"
      )
    ) |>
    dplyr::group_by(
      scenario,
      n,
      p,
      p_over_n,
      method
    ) |>
    dplyr::summarise(
      B =
        dplyr::n(),
      mean_selected_alpha =
        if (
          all(
            is.na(
              selected_alpha
            )
          )
        ) {
          NA_real_
        } else {
          mean(
            selected_alpha,
            na.rm = TRUE
          )
        },
      mean_selected_mixing =
        if (
          all(
            is.na(
              selected_mixing
            )
          )
        ) {
          NA_real_
        } else {
          mean(
            selected_mixing,
            na.rm = TRUE
          )
        },
      median_selected_lambda =
        if (
          all(
            is.na(
              selected_lambda
            )
          )
        ) {
          NA_real_
        } else {
          stats::median(
            selected_lambda,
            na.rm = TRUE
          )
        },
      lambda_boundary_frequency =
        if (
          all(
            is.na(
              lambda_boundary
            )
          )
        ) {
          NA_real_
        } else {
          mean(
            lambda_boundary,
            na.rm = TRUE
          )
        },
      .groups = "drop"
    )
}


# ============================================================
# 15.7 PAIRED BENCHMARK-VERSUS-NEURAL CONTRASTS
#
# The linear benchmark observation is duplicated only at this
# analysis stage across L in {1,3,5}; it is NOT refitted.
#
# Difference is always:
#
#   first method - neural reference.
# ============================================================

make_benchmark_neural_contrasts <- function(
    raw
) {

  neural <-
    raw |>
    dplyr::filter(
      method_group == "neural"
    ) |>
    dplyr::select(
      scenario,
      n,
      p,
      p_over_n,
      depth,
      rep,
      neural_method = method,
      neural_label =
        method_label,
      neural_risk = risk,
      neural_conditional_risk =
        conditional_risk,
      neural_auc = auc,
      neural_brier = brier,
      neural_logistic_risk =
        logistic_risk
    )

  benchmark <-
    raw |>
    dplyr::filter(
      method_group == "linear"
    ) |>
    dplyr::select(
      scenario,
      n,
      p,
      p_over_n,
      rep,
      benchmark_method = method,
      benchmark_label =
        method_label,
      benchmark_risk = risk,
      benchmark_conditional_risk =
        conditional_risk,
      benchmark_auc = auc,
      benchmark_brier = brier,
      benchmark_logistic_risk =
        logistic_risk
    )

  benchmark |>
    dplyr::inner_join(
      neural,
      by = c(
        "scenario",
        "n",
        "p",
        "p_over_n",
        "rep"
      )
    ) |>
    dplyr::transmute(
      scenario,
      n,
      p,
      p_over_n,
      depth,
      rep,
      benchmark_method,
      benchmark_label,
      neural_method,
      neural_label,
      contrast =
        paste0(
          benchmark_label,
          " - ",
          neural_label
        ),
      risk_difference =
        benchmark_risk -
        neural_risk,
      conditional_risk_difference =
        benchmark_conditional_risk -
        neural_conditional_risk,
      auc_difference =
        benchmark_auc -
        neural_auc,
      brier_difference =
        benchmark_brier -
        neural_brier,
      logistic_risk_difference =
        benchmark_logistic_risk -
        neural_logistic_risk
    )
}


# ============================================================
# 15.8 VALIDATE COMPLETED PRODUCTION OUTPUT
# ============================================================

validate_production_results <- function(
    compiled,
    spec =
      final_simulation_spec
) {

  raw <-
    compiled$raw

  design <-
    make_production_design(
      spec
    )

  expected_neural_cells <-
    nrow(
      design$neural
    )

  expected_linear_cells <-
    nrow(
      design$linear
    )

  observed_neural_cells <-
    raw |>
    dplyr::filter(
      method_group == "neural"
    ) |>
    dplyr::distinct(
      scenario,
      p_over_n,
      depth
    ) |>
    nrow()

  observed_linear_cells <-
    raw |>
    dplyr::filter(
      method_group == "linear"
    ) |>
    dplyr::distinct(
      scenario,
      p_over_n
    ) |>
    nrow()

  replicate_counts <-
    raw |>
    dplyr::group_by(
      scenario,
      p_over_n,
      depth,
      method
    ) |>
    dplyr::summarise(
      B =
        dplyr::n(),
      .groups = "drop"
    )

  # For linear methods depth is NA, so this still gives one
  # count per scenario x p/n x method.
  incomplete_groups <-
    replicate_counts |>
    dplyr::filter(
      B != spec$B
    )

  endpoint_failures <-
    raw |>
    dplyr::filter(
      method == "CR-DNN",
      selected_alpha == 1,
      (
        is.na(
          endpoint_probability_difference
        ) |
        endpoint_probability_difference >
        1e-10
      )
    )

  list(
    expected_neural_cells =
      expected_neural_cells,
    observed_neural_cells =
      observed_neural_cells,
    expected_linear_cells =
      expected_linear_cells,
    observed_linear_cells =
      observed_linear_cells,
    incomplete_groups =
      incomplete_groups,
    endpoint_failures =
      endpoint_failures,
    complete =
      observed_neural_cells ==
      expected_neural_cells &&
      observed_linear_cells ==
      expected_linear_cells &&
      nrow(
        incomplete_groups
      ) == 0L &&
      nrow(
        endpoint_failures
      ) == 0L
  )
}


# ============================================================
# 15.9 WRITE MANUSCRIPT-READY SUMMARY TABLES
# ============================================================

write_production_summaries <- function(
    compiled,
    output_dir =
      file.path(
        "results",
        "compiled"
      )
) {

  dir.create(
    output_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  method_summary <-
    summarise_method_performance(
      compiled$raw
    )

  neural_summary <-
    summarise_neural_contrasts(
      compiled$
      neural_contrasts
    )

  cr_selection <-
    summarise_crdnn_selection(
      compiled$
      selection
    )

  linear_selection <-
    summarise_linear_selection(
      compiled$
      selection
    )

  benchmark_contrasts <-
    make_benchmark_neural_contrasts(
      compiled$raw
    )

  write.csv(
    method_summary,
    file.path(
      output_dir,
      "table_method_performance.csv"
    ),
    row.names = FALSE
  )

  write.csv(
    neural_summary,
    file.path(
      output_dir,
      "table_crdnn_vs_dnn.csv"
    ),
    row.names = FALSE
  )

  write.csv(
    cr_selection,
    file.path(
      output_dir,
      "table_crdnn_alpha_selection.csv"
    ),
    row.names = FALSE
  )

  write.csv(
    linear_selection,
    file.path(
      output_dir,
      "table_linear_tuning.csv"
    ),
    row.names = FALSE
  )

  write.csv(
    benchmark_contrasts,
    file.path(
      output_dir,
      "paired_benchmark_neural_contrasts.csv"
    ),
    row.names = FALSE
  )

  invisible(
    list(
      method_summary =
        method_summary,
      neural_summary =
        neural_summary,
      cr_selection =
        cr_selection,
      linear_selection =
        linear_selection,
      benchmark_contrasts =
        benchmark_contrasts
    )
  )
}


# ============================================================
# 16. PRODUCTION EXECUTION EXAMPLES
#
# DO NOT run all cells interactively by accident.
#
# HPC neural array:
#   cell_id in 1:54
#
# HPC linear array:
#   cell_id in 1:18
# ============================================================

production_design <-
  make_production_design()

# Avoid flooding HPC logs every time the production file is sourced.
# Set CRDNN_PRINT_DESIGN=1 to print the tables in a batch session.
if (
  interactive() ||
  identical(
    Sys.getenv("CRDNN_PRINT_DESIGN", "0"),
    "1"
  )
) {
  print(
    production_design$neural,
    n = Inf
  )

  print(
    production_design$linear,
    n = Inf
  )
}


# Example only:
# 
# run_neural_production_cell(
#   cell_id = 1L,
#   results_dir = "results"
# )
# 
# run_linear_production_cell(
#   cell_id = 1L,
#   results_dir = "results"
# )

#
# After all jobs finish:
#
# compiled <-
#   compile_production_results(
#     results_dir = "results"
#   )
#
# validation <-
#   validate_production_results(
#     compiled
#   )
#
# print(validation)
#
# if (!validation$complete) {
#   stop("Production results are incomplete or failed invariants.")
# }
#
# final_tables <-
#   write_production_summaries(
#     compiled
#   )



# ============================================================
# 17. HIGH-THROUGHPUT NEURAL PRODUCTION OVERRIDES
#
# These functions preserve the frozen statistical design while
# eliminating repeated R -> TensorFlow copies and enabling one
# Monte Carlo replication per SLURM array task.
#
# The original cell-level functions remain available.
# ============================================================


# ============================================================
# 17.1 BACKEND TENSOR HELPERS
# ============================================================

crdnn_float32_tensor <- function(x) {
  keras3::op_convert_to_tensor(
    x,
    dtype = "float32"
  )
}


crdnn_binary_tensor <- function(y) {

  y <- as.numeric(y)

  crdnn_float32_tensor(
    array(
      y,
      dim = c(
        length(y),
        1L
      )
    )
  )
}


# ============================================================
# 17.1A CACHE ALPHA-INVARIANT SHRINKAGE PROJECTIONS
#
# For fixed training covariance preparation and a fixed data matrix,
# both Xc and Xc %*% V are independent of alpha.  The original
# apply_shrinkage() recomputes the latter for every alpha.  Cache it
# once and vary only the diagonal shrinkage coefficients.
# ============================================================

prepare_shrinkage_application_cache <- function(
    prep,
    X
) {

  X <- as.matrix(X)

  if (ncol(X) != prep$p) {
    stop(
      "X must have the same number of predictors as prep.",
      call. = FALSE
    )
  }

  Xc <-
    sweep(
      X,
      2L,
      prep$xbar,
      FUN = "-"
    )

  projected <-
    if (length(prep$eigvals) == 0L) {
      NULL
    } else {
      Xc %*%
        prep$V
    }

  list(
    Xc = Xc,
    projected = projected
  )
}


apply_shrinkage_cached <- function(
    prep,
    cache,
    alpha
) {

  alpha <- as.numeric(alpha)

  if (
    length(alpha) != 1L ||
    !is.finite(alpha) ||
    alpha <= 0 ||
    alpha > 1
  ) {
    stop(
      "alpha must satisfy 0 < alpha <= 1.",
      call. = FALSE
    )
  }

  # Exact identity endpoint.  At alpha = 1 the scale-preserving
  # CR-DNN transformation is precisely (X - xbar) / sqrt(p), so
  # no eigenspace back-projection is required.
  if (isTRUE(all.equal(alpha, 1))) {
    return(
      list(
        Z = cache$Xc / sqrt(prep$p),
        scale_match = sqrt(prep$tau_cov),
        transformed_variance = prep$tau_cov,
        condition_number = 1
      )
    )
  }

  c0 <-
    alpha *
    prep$tau_cov

  if (!is.finite(c0) || c0 <= 0) {
    stop(
      "alpha * tau_cov must be positive.",
      call. = FALSE
    )
  }

  base_scale <-
    1 /
    sqrt(c0)

  if (length(prep$eigvals) == 0L) {

    Z_raw <-
      base_scale *
      cache$Xc

    whitened_variance <- 0

  } else {

    shrunk_eigenvalues <-
      c0 +
      (1 - alpha) *
      prep$eigvals

    adjustment <-
      1 /
      sqrt(shrunk_eigenvalues) -
      base_scale

    Z_raw <-
      base_scale *
      cache$Xc +
      sweep(
        cache$projected,
        2L,
        adjustment,
        FUN = "*"
      ) %*%
      t(prep$V)

    whitened_variance <-
      sum(
        prep$eigvals /
        shrunk_eigenvalues
      ) /
      prep$p
  }

  if (
    !is.finite(whitened_variance) ||
    whitened_variance <= 0
  ) {
    stop(
      "Invalid transformed empirical variance.",
      call. = FALSE
    )
  }

  scale_match <-
    sqrt(
      prep$tau_cov /
      whitened_variance
    )

  Z <-
    scale_match *
    Z_raw /
    sqrt(prep$p)

  list(
    Z = Z,
    scale_match = scale_match,
    transformed_variance =
      whitened_variance *
      scale_match^2,
    condition_number =
      shrinkage_condition_number(
        eigvals = prep$eigvals,
        p = prep$p,
        alpha = alpha,
        tau_cov = prep$tau_cov
      )
  )
}


# ============================================================
# 17.2 FAST CV FIT ON PRE-CONVERTED TENSORS
#
# X/y are already backend-native float32 tensors.  Hence the same
# transformed fold is transferred from R to TensorFlow only once
# per alpha, rather than once for every lambda candidate.
# ============================================================

fit_predict_nn_cv_fast <- function(
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    input_dim,
    depth,
    width,
    lambda_nn,
    penalty_tau = 1,
    learning_rate = 1e-3,
    epochs = 100L,
    batch_size = 32L,
    validation_split = 0.10,
    patience = 10L,
    seed = 1L
) {

  # Clear graph bookkeeping but avoid Python garbage collection on
  # every candidate.  This is materially cheaper in large grids.
  try(
    keras3::clear_session(
      free_memory = FALSE
    ),
    silent = TRUE
  )

  keras3::set_random_seed(
    as.integer(seed)
  )

  model <-
    build_nn(
      input_dim = as.integer(input_dim),
      depth = as.integer(depth),
      width = as.integer(width),
      lambda_nn = as.numeric(lambda_nn),
      penalty_tau = as.numeric(penalty_tau),
      learning_rate = as.numeric(learning_rate)
    )

  early_stop <-
    keras3::callback_early_stopping(
      monitor = "val_binary_crossentropy",
      mode = "min",
      patience = as.integer(patience),
      restore_best_weights = TRUE
    )

  history <-
    generics::fit(
      model,
      x = X_train_tensor,
      y = y_train_tensor,
      epochs = as.integer(epochs),
      batch_size = as.integer(batch_size),
      validation_split = as.numeric(validation_split),
      validation_batch_size =
        as.integer(
          max(
            batch_size,
            256L
          )
        ),
      shuffle = TRUE,
      callbacks = list(early_stop),
      verbose = 0L,
      view_metrics = FALSE
    )

  val_bce <-
    as.numeric(
      history$metrics$
      val_binary_crossentropy
    )

  if (
    length(val_bce) == 0L ||
    !any(is.finite(val_bce))
  ) {
    stop(
      "Fast CV fit did not record val_binary_crossentropy.",
      call. = FALSE
    )
  }

  prob <-
    as.numeric(
      as.array(
        model(
          X_test_tensor,
          training = FALSE
        )
      )
    )

  prob <- pmin(
    pmax(prob, 0),
    1
  )

  list(
    prob = prob,
    epochs_used = as.integer(length(val_bce)),
    best_epoch = as.integer(which.min(val_bce)),
    best_inner_val_bce = min(val_bce, na.rm = TRUE)
  )
}


# ============================================================
# 17.3 TENSOR-CACHED JOINT CR-DNN CV
#
# Statistical design is unchanged:
#   * same outer folds;
#   * same alpha grid;
#   * same lambda grid;
#   * same inner validation fraction;
#   * same candidate seed within a fold;
#   * same pooled OOF BCE selection rule.
#
# Computational change:
#   * each alpha-specific fit/validation representation is copied
#     to the TensorFlow backend ONCE and reused for every lambda.
# ============================================================

tune_crdnn_cv <- function(
    X_train,
    y_train,
    depth,
    width,
    alpha_grid = c(
      0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00
    ),
    lambda_nn_grid = c(
      0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
    ),
    foldid = NULL,
    nfolds = 5L,
    inner_validation_fraction = 0.10,
    penalty_tau = 1,
    learning_rate = 1e-3,
    epochs = 100L,
    batch_size = 32L,
    patience = 10L,
    seed = 1L
) {

  X_train <- as.matrix(X_train)
  y_train <- as.integer(y_train)
  n <- nrow(X_train)

  if (length(y_train) != n) {
    stop(
      "nrow(X_train) must equal length(y_train).",
      call. = FALSE
    )
  }

  if (
    any(!y_train %in% c(0L, 1L)) ||
    length(unique(y_train)) < 2L
  ) {
    stop(
      "y_train must contain both binary classes 0/1.",
      call. = FALSE
    )
  }

  if (any(!is.finite(X_train))) {
    stop(
      "X_train contains non-finite values.",
      call. = FALSE
    )
  }

  alpha_grid <-
    sort(
      unique(
        as.numeric(alpha_grid)
      )
    )

  lambda_nn_grid <-
    sort(
      unique(
        as.numeric(lambda_nn_grid)
      )
    )

  if (
    any(!is.finite(alpha_grid)) ||
    any(alpha_grid <= 0 | alpha_grid > 1)
  ) {
    stop(
      "alpha_grid must satisfy 0 < alpha <= 1.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(lambda_nn_grid)) ||
    any(lambda_nn_grid < 0)
  ) {
    stop(
      "lambda_nn_grid must be finite and non-negative.",
      call. = FALSE
    )
  }

  if (is.null(foldid)) {
    foldid <-
      make_stratified_foldid(
        y = y_train,
        k = nfolds,
        seed = seed
      )
  }

  foldid <- as.integer(foldid)

  if (
    length(foldid) != n ||
    any(!is.finite(foldid))
  ) {
    stop(
      "Invalid foldid.",
      call. = FALSE
    )
  }

  fold_levels <-
    sort(
      unique(foldid)
    )

  K <- length(fold_levels)

  if (K < 3L) {
    stop(
      "At least three outer folds are required.",
      call. = FALSE
    )
  }

  candidate_grid <-
    tidyr::expand_grid(
      alpha = alpha_grid,
      lambda_nn = lambda_nn_grid
    ) |>
    dplyr::mutate(
      candidate_id =
        dplyr::row_number()
    )

  n_alpha <- length(alpha_grid)
  n_lambda <- length(lambda_nn_grid)
  n_candidates <- n_alpha * n_lambda

  oof_prob <-
    matrix(
      NA_real_,
      nrow = n,
      ncol = n_candidates
    )

  oof_null_prob <-
    rep(
      NA_real_,
      n
    )

  epochs_matrix <-
    matrix(
      NA_real_,
      nrow = K,
      ncol = n_candidates
    )

  best_epoch_matrix <-
    matrix(
      NA_real_,
      nrow = K,
      ncol = n_candidates
    )

  condition_matrix <-
    matrix(
      NA_real_,
      nrow = K,
      ncol = n_alpha
    )

  for (kk in seq_along(fold_levels)) {

    fold_k <- fold_levels[kk]
    val_idx <- which(foldid == fold_k)
    fit_idx <- which(foldid != fold_k)

    X_fit_raw <-
      X_train[
        fit_idx,
        ,
        drop = FALSE
      ]

    X_val_raw <-
      X_train[
        val_idx,
        ,
        drop = FALSE
      ]

    y_fit <- y_train[fit_idx]

    if (length(unique(y_fit)) < 2L) {
      stop(
        "Each outer fitting fold must contain both classes.",
        call. = FALSE
      )
    }

    oof_null_prob[val_idx] <- mean(y_fit)

    shrink_prep <-
      prepare_shrinkage(
        X_fit_raw
      )

    fit_transform_cache <-
      prepare_shrinkage_application_cache(
        prep = shrink_prep,
        X = X_fit_raw
      )

    val_transform_cache <-
      prepare_shrinkage_application_cache(
        prep = shrink_prep,
        X = X_val_raw
      )

    # One target tensor per outer fold, shared across every alpha
    # and lambda candidate.
    y_fit_tensor <-
      crdnn_binary_tensor(
        y_fit
      )

    fold_seed <-
      seed +
      10000L * kk

    for (aa in seq_along(alpha_grid)) {

      alpha_a <- alpha_grid[aa]

      fit_white <-
        apply_shrinkage_cached(
          prep = shrink_prep,
          cache = fit_transform_cache,
          alpha = alpha_a
        )

      val_white <-
        apply_shrinkage_cached(
          prep = shrink_prep,
          cache = val_transform_cache,
          alpha = alpha_a
        )

      condition_matrix[kk, aa] <-
        fit_white$condition_number

      # Critical optimisation: transfer each alpha representation
      # once, then reuse it for all lambda candidates.
      X_fit_tensor <-
        crdnn_float32_tensor(
          fit_white$Z
        )

      X_val_tensor <-
        crdnn_float32_tensor(
          val_white$Z
        )

      rm(
        fit_white,
        val_white
      )

      for (ll in seq_along(lambda_nn_grid)) {

        lambda_l <- lambda_nn_grid[ll]

        # expand_grid() varies lambda within alpha, so this is the
        # exact candidate row without an inner-loop dplyr filter.
        candidate_id <-
          (aa - 1L) *
          n_lambda +
          ll

        fit_l <-
          fit_predict_nn_cv_fast(
            X_train_tensor = X_fit_tensor,
            y_train_tensor = y_fit_tensor,
            X_test_tensor = X_val_tensor,
            input_dim = ncol(X_fit_raw),
            depth = depth,
            width = width,
            lambda_nn = lambda_l,
            penalty_tau = penalty_tau,
            learning_rate = learning_rate,
            epochs = epochs,
            batch_size = batch_size,
            validation_split =
              inner_validation_fraction,
            patience = patience,
            seed = fold_seed
          )

        oof_prob[val_idx, candidate_id] <-
          fit_l$prob

        epochs_matrix[kk, candidate_id] <-
          fit_l$epochs_used

        best_epoch_matrix[kk, candidate_id] <-
          fit_l$best_epoch
      }

      rm(
        X_fit_tensor,
        X_val_tensor
      )
    }

    rm(
      y_fit_tensor,
      fit_transform_cache,
      val_transform_cache,
      shrink_prep
    )
  }

  if (any(!is.finite(oof_prob))) {
    stop(
      "Incomplete CR-DNN out-of-fold predictions.",
      call. = FALSE
    )
  }

  if (any(!is.finite(oof_null_prob))) {
    stop(
      "Incomplete null-model out-of-fold predictions.",
      call. = FALSE
    )
  }

  null_cv_loss <-
    binary_log_loss(
      y = y_train,
      prob = oof_null_prob
    )

  cv_validation_loss <-
    vapply(
      seq_len(n_candidates),
      function(cc) {
        binary_log_loss(
          y_train,
          oof_prob[, cc]
        )
      },
      numeric(1)
    )

  cv_auc <-
    vapply(
      seq_len(n_candidates),
      function(cc) {
        safe_auc(
          y_train,
          oof_prob[, cc]
        )
      },
      numeric(1)
    )

  cv_brier <-
    colMeans(
      (
        oof_prob -
        y_train
      )^2
    )

  alpha_index <-
    match(
      candidate_grid$alpha,
      alpha_grid
    )

  tuning_table <-
    candidate_grid |>
    dplyr::mutate(
      cv_validation_loss =
        cv_validation_loss,
      null_cv_loss =
        null_cv_loss,
      cv_loss_gain =
        null_cv_loss -
        cv_validation_loss,
      cv_auc = cv_auc,
      cv_brier = cv_brier,
      cv_prob_sd =
        apply(
          oof_prob,
          2L,
          stats::sd
        ),
      cv_prob_min =
        apply(
          oof_prob,
          2L,
          min
        ),
      cv_prob_max =
        apply(
          oof_prob,
          2L,
          max
        ),
      mean_epochs_used =
        colMeans(
          epochs_matrix,
          na.rm = TRUE
        ),
      mean_best_epoch =
        colMeans(
          best_epoch_matrix,
          na.rm = TRUE
        ),
      mean_condition_number =
        colMeans(
          condition_matrix,
          na.rm = TRUE
        )[alpha_index],
      outer_folds = K
    ) |>
    dplyr::select(
      -candidate_id
    ) |>
    dplyr::arrange(
      cv_validation_loss,
      alpha,
      lambda_nn
    )

  list(
    selected_alpha =
      tuning_table$alpha[1L],
    selected_lambda =
      tuning_table$lambda_nn[1L],
    tuning_table = tuning_table,
    foldid = foldid,
    candidate_grid = candidate_grid,
    oof_prob = oof_prob,
    oof_null_prob = oof_null_prob
  )
}


# ============================================================
# 17.4 REPLICATION-LEVEL HPC TASK MAP
#
# task_id = 1,...,(54 * B) by default.
#
# This is the key wall-clock optimisation.  The original program
# executes B=20 replications serially inside each of 54 jobs.
# Here each replication is independently schedulable, exposing
# all Monte Carlo parallelism to SLURM.
# ============================================================

make_neural_replication_design <- function(
    spec = final_simulation_spec,
    reps_per_task = 2L
) {

  reps_per_task <-
    as.integer(
      reps_per_task
    )

  if (
    length(reps_per_task) != 1L ||
    is.na(reps_per_task) ||
    reps_per_task < 1L
  ) {
    stop(
      "reps_per_task must be a positive integer.",
      call. = FALSE
    )
  }

  cells <-
    make_production_design(
      spec
    )$neural

  B <- as.integer(spec$B)

  chunks_per_cell <-
    as.integer(
      ceiling(
        B /
        reps_per_task
      )
    )

  design <-
    tidyr::crossing(
      cell_id = cells$cell_id,
      chunk_id = seq_len(
        chunks_per_cell
      )
    ) |>
    dplyr::left_join(
      cells,
      by = "cell_id"
    ) |>
    dplyr::mutate(
      rep_start =
        (chunk_id - 1L) *
        reps_per_task +
        1L,
      rep_end =
        pmin(
          chunk_id *
            reps_per_task,
          B
        ),
      task_id =
        dplyr::row_number()
    ) |>
    dplyr::select(
      task_id,
      dplyr::everything()
    )

  design
}


# ============================================================
# 17.5 SAVE ONE NEURAL REPLICATION
# ============================================================

save_neural_replication <- function(
    result,
    scenario,
    p_over_n,
    depth,
    rep_ids,
    results_dir = "results_fast"
) {

  dir.create(
    results_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  rep_ids <- as.integer(rep_ids)

  base_stub <-
    make_cell_stub(
      layer = "neural",
      scenario = scenario,
      p_over_n = p_over_n,
      depth = depth
    )

  rep_text <-
    if (length(rep_ids) == 1L) {
      sprintf(
        "%02d",
        rep_ids[1L]
      )
    } else {
      paste0(
        sprintf(
          "%02d",
          min(rep_ids)
        ),
        "_",
        sprintf(
          "%02d",
          max(rep_ids)
        )
      )
    }

  stub <-
    paste0(
      base_stub,
      "__rep_",
      rep_text
    )

  # One checkpoint file per task is substantially kinder to a
  # shared HPC filesystem than multiple CSV files per replication.
  # Compression is disabled because these are temporary production
  # checkpoints and write/read speed is more important than size.
  saveRDS(
    result,
    file = file.path(
      results_dir,
      paste0(
        stub,
        ".rds"
      )
    ),
    compress = FALSE
  )

  invisible(stub)
}


# ============================================================
# 17.6 RUN ONE REPLICATION-LEVEL ARRAY TASK
# ============================================================

run_neural_replication_task <- function(
    task_id,
    results_dir = "results_fast",
    overwrite = FALSE,
    spec = final_simulation_spec,
    reps_per_task = 2L,
    verbose = TRUE
) {

  design <-
    make_neural_replication_design(
      spec = spec,
      reps_per_task = reps_per_task
    )

  row <-
    design[
      design$task_id ==
        as.integer(task_id),
      ,
      drop = FALSE
    ]

  if (nrow(row) != 1L) {
    stop(
      "task_id must identify exactly one neural replication chunk.",
      call. = FALSE
    )
  }

  rep_ids <-
    seq.int(
      from = as.integer(row$rep_start),
      to = as.integer(row$rep_end)
    )

  base_stub <-
    make_cell_stub(
      layer = "neural",
      scenario = row$scenario,
      p_over_n = row$p_over_n,
      depth = row$depth
    )

  rep_text <-
    if (length(rep_ids) == 1L) {
      sprintf(
        "%02d",
        rep_ids[1L]
      )
    } else {
      paste0(
        sprintf(
          "%02d",
          min(rep_ids)
        ),
        "_",
        sprintf(
          "%02d",
          max(rep_ids)
        )
      )
    }

  checkpoint_file <-
    file.path(
      results_dir,
      paste0(
        base_stub,
        "__rep_",
        rep_text,
        ".rds"
      )
    )

  if (
    file.exists(checkpoint_file) &&
    !overwrite
  ) {
    message(
      "Skipping completed neural task: ",
      task_id
    )
    return(
      invisible(NULL)
    )
  }

  result <-
    run_neural_mc_cell(
      scenario = row$scenario,
      n = row$n,
      p_over_n = row$p_over_n,
      depth = row$depth,
      B = as.integer(spec$B),
      rep_ids = rep_ids,
      n_test = row$n_test,
      master_seed = spec$master_seed,
      spec = spec,
      verbose = verbose
    )

  save_neural_replication(
    result = result,
    scenario = row$scenario,
    p_over_n = row$p_over_n,
    depth = row$depth,
    rep_ids = rep_ids,
    results_dir = results_dir
  )

  invisible(result)
}


# ============================================================
# 17.7 COMPILE FAST REPLICATION OUTPUT
#
# Use a fresh results directory for the fast neural run so that
# old cell-level neural files are not double-counted.
# ============================================================

compile_fast_neural_results <- function(
    results_dir = "results_fast",
    compiled_dir =
      file.path(
        results_dir,
        "compiled"
      ),
    write_output = TRUE
) {

  checkpoint_files <-
    list.files(
      results_dir,
      pattern =
        "^neural__.*__rep_[0-9]+(_[0-9]+)?\\.rds$",
      full.names = TRUE
    )

  if (length(checkpoint_files) == 0L) {
    stop(
      "No fast neural replication checkpoints were found.",
      call. = FALSE
    )
  }

  objects <-
    lapply(
      checkpoint_files,
      readRDS
    )

  bind_component <- function(name) {
    dplyr::bind_rows(
      lapply(
        objects,
        function(x) {
          x[[name]]
        }
      )
    )
  }

  raw <- bind_component("raw")
  selection <- bind_component("selection")
  tuning <- bind_component("tuning")
  invariants <- bind_component("invariants")
  contrasts <- bind_component("contrasts")

  dgp <-
    bind_component("dgp") |>
    dplyr::distinct(
      scenario,
      n,
      p,
      p_over_n,
      depth,
      .keep_all = TRUE
    )

  compiled <-
    list(
      raw = raw,
      selection = selection,
      tuning = tuning,
      invariants = invariants,
      dgp = dgp,
      neural_contrasts = contrasts
    )

  if (write_output) {

    dir.create(
      compiled_dir,
      recursive = TRUE,
      showWarnings = FALSE
    )

    write.csv(
      raw,
      file.path(
        compiled_dir,
        "production_raw.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      selection,
      file.path(
        compiled_dir,
        "production_selection.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      tuning,
      file.path(
        compiled_dir,
        "production_tuning.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      invariants,
      file.path(
        compiled_dir,
        "production_invariants.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      dgp,
      file.path(
        compiled_dir,
        "production_dgp.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      contrasts,
      file.path(
        compiled_dir,
        "production_neural_contrasts.csv"
      ),
      row.names = FALSE
    )
  }

  compiled
}


# ============================================================
# 17.8 FAST-RUN INFORMATION
# ============================================================

crdnn_fast_run_info <- function(
    spec = final_simulation_spec,
    reps_per_task = 2L
) {

  design <-
    make_neural_replication_design(
      spec = spec,
      reps_per_task = reps_per_task
    )

  cv_fits_per_replication <-
    length(spec$alpha_grid) *
    length(spec$lambda_nn_grid) *
    as.integer(spec$outer_folds)

  tibble::tibble(
    neural_cells =
      nrow(
        make_production_design(
          spec
        )$neural
      ),
    replications_per_cell =
      as.integer(spec$B),
    reps_per_task =
      as.integer(reps_per_task),
    array_tasks =
      nrow(design),
    cv_fits_per_replication =
      cv_fits_per_replication,
    fits_per_replication =
      cv_fits_per_replication +
      2L,
    approximate_fits_per_task =
      as.integer(reps_per_task) *
      (
        cv_fits_per_replication +
        2L
      ),
    note =
      paste0(
        "Chunked Monte Carlo tasks; statistical design unchanged."
      )
  )
}


# ============================================================
# 17.9 MERGE FAST NEURAL RESULTS WITH EXISTING LINEAR RESULTS
#
# This avoids rerunning the completed linear benchmark layer.
# The linear directory may also contain partial neural files;
# those are explicitly filtered out.
# ============================================================

compile_fast_with_existing_linear <- function(
    neural_results_dir = "results_fast",
    linear_results_dir = "results",
    compiled_dir =
      file.path(
        neural_results_dir,
        "compiled_full"
      ),
    write_output = TRUE
) {

  neural <-
    compile_fast_neural_results(
      results_dir = neural_results_dir,
      write_output = FALSE
    )

  linear_raw <-
    read_result_family(
      linear_results_dir,
      "__raw"
    ) |>
    dplyr::filter(
      method_group == "linear"
    )

  if (nrow(linear_raw) == 0L) {
    stop(
      "No existing linear benchmark results were found.",
      call. = FALSE
    )
  }

  linear_selection <-
    read_result_family(
      linear_results_dir,
      "__selection"
    )

  if (
    nrow(linear_selection) > 0L &&
    "method" %in%
      names(linear_selection)
  ) {
    linear_selection <-
      linear_selection |>
      dplyr::filter(
        method %in%
          c(
            "ShrinkageLDA",
            "RidgeLogit",
            "ElasticNetLogit"
          )
      )
  }

  linear_tuning <-
    read_result_family(
      linear_results_dir,
      "__tuning"
    )

  if (
    nrow(linear_tuning) > 0L &&
    "method" %in%
      names(linear_tuning)
  ) {
    linear_tuning <-
      linear_tuning |>
      dplyr::filter(
        method %in%
          c(
            "ShrinkageLDA",
            "RidgeLogit",
            "ElasticNetLogit"
          )
      )
  }

  linear_invariants <-
    read_result_family(
      linear_results_dir,
      "__invariants"
    )

  linear_dgp <-
    read_result_family(
      linear_results_dir,
      "__dgp"
    )

  compiled <-
    list(
      raw =
        dplyr::bind_rows(
          neural$raw,
          linear_raw
        ),
      selection =
        dplyr::bind_rows(
          neural$selection,
          linear_selection
        ),
      tuning =
        dplyr::bind_rows(
          neural$tuning,
          linear_tuning
        ),
      invariants =
        dplyr::bind_rows(
          neural$invariants,
          linear_invariants
        ),
      dgp =
        dplyr::bind_rows(
          neural$dgp,
          linear_dgp
        ) |>
        dplyr::distinct(),
      neural_contrasts =
        neural$neural_contrasts
    )

  if (write_output) {

    dir.create(
      compiled_dir,
      recursive = TRUE,
      showWarnings = FALSE
    )

    write.csv(
      compiled$raw,
      file.path(
        compiled_dir,
        "production_raw.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      compiled$selection,
      file.path(
        compiled_dir,
        "production_selection.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      compiled$tuning,
      file.path(
        compiled_dir,
        "production_tuning.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      compiled$invariants,
      file.path(
        compiled_dir,
        "production_invariants.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      compiled$dgp,
      file.path(
        compiled_dir,
        "production_dgp.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      compiled$neural_contrasts,
      file.path(
        compiled_dir,
        "production_neural_contrasts.csv"
      ),
      row.names = FALSE
    )
  }

  compiled
}

# ============================================================
# 18.0 FROZEN NEURAL TUNING-GRID VALIDATION
#
# Prevents a sequential production launch with a stale tuning grid.
# ============================================================

validate_frozen_neural_grid <- function(spec = final_simulation_spec) {

  expected_alpha <- c(
    0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00
  )

  expected_lambda <- c(
    0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
  )

  if (!isTRUE(all.equal(
    as.numeric(spec$alpha_grid),
    expected_alpha,
    tolerance = 0,
    check.attributes = FALSE
  ))) {
    stop(
      "Sequential neural production requires the frozen 8-value alpha grid.",
      call. = FALSE
    )
  }

  if (!isTRUE(all.equal(
    as.numeric(spec$lambda_nn_grid),
    expected_lambda,
    tolerance = 0,
    check.attributes = FALSE
  ))) {
    stop(
      "Sequential neural production requires the frozen 9-value lambda grid.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# ============================================================
# 18.1 SEQUENTIAL MONTE CARLO PRECISION STOPPING
#
# The statistical simulation itself is unchanged.  This section
# changes only the allocation of Monte Carlo replications across
# neural design cells.  Stopping depends on Monte Carlo precision
# (dispersion of paired CR-DNN minus DNN contrasts), never on the
# sign, magnitude, or statistical significance of the estimated
# performance advantage.
# ============================================================

sequential_mc_spec <- list(
  B_min = 12L,
  B_max = 20L,
  batch_reps = 2L,
  confidence_level = 0.95,
  conditional_risk_halfwidth = 0.010,
  auc_halfwidth = 0.020
)

validate_sequential_mc_spec <- function(seq_spec = sequential_mc_spec) {
  required <- c(
    "B_min", "B_max", "batch_reps", "confidence_level",
    "conditional_risk_halfwidth", "auc_halfwidth"
  )
  missing <- setdiff(required, names(seq_spec))
  if (length(missing) > 0L) {
    stop("Missing sequential MC controls: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  B_min <- as.integer(seq_spec$B_min)
  B_max <- as.integer(seq_spec$B_max)
  batch_reps <- as.integer(seq_spec$batch_reps)

  if (B_min < 2L || B_max < B_min || batch_reps < 1L) {
    stop("Require 2 <= B_min <= B_max and batch_reps >= 1.", call. = FALSE)
  }
  if (!is.finite(seq_spec$confidence_level) ||
      seq_spec$confidence_level <= 0 || seq_spec$confidence_level >= 1) {
    stop("confidence_level must lie strictly between 0 and 1.", call. = FALSE)
  }
  if (!is.finite(seq_spec$conditional_risk_halfwidth) ||
      seq_spec$conditional_risk_halfwidth <= 0 ||
      !is.finite(seq_spec$auc_halfwidth) || seq_spec$auc_halfwidth <= 0) {
    stop("Precision half-width tolerances must be positive finite values.", call. = FALSE)
  }

  invisible(TRUE)
}

summarise_sequential_neural_precision <- function(
    neural_contrasts,
    seq_spec = sequential_mc_spec
) {
  validate_sequential_mc_spec(seq_spec)

  if (nrow(neural_contrasts) == 0L) {
    return(tibble::tibble())
  }

  required <- c(
    "scenario", "n", "p", "p_over_n", "depth", "rep",
    "conditional_risk_difference", "auc_difference"
  )
  missing <- setdiff(required, names(neural_contrasts))
  if (length(missing) > 0L) {
    stop("Missing neural contrast columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  level <- as.numeric(seq_spec$confidence_level)

  neural_contrasts |>
    dplyr::group_by(
      scenario, n, p, p_over_n, depth
    ) |>
    dplyr::summarise(
      B = dplyr::n_distinct(rep),
      min_rep = min(rep),
      max_rep = max(rep),
      mean_conditional_risk_difference =
        mean(conditional_risk_difference, na.rm = TRUE),
      sd_conditional_risk_difference =
        stats::sd(conditional_risk_difference, na.rm = TRUE),
      mean_auc_difference =
        mean(auc_difference, na.rm = TRUE),
      sd_auc_difference =
        stats::sd(auc_difference, na.rm = TRUE),
      finite_conditional_risk =
        sum(is.finite(conditional_risk_difference)),
      finite_auc =
        sum(is.finite(auc_difference)),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      rep_sequence_complete =
        min_rep == 1L & B == max_rep,
      mcse_conditional_risk =
        sd_conditional_risk_difference / sqrt(B),
      mcse_auc =
        sd_auc_difference / sqrt(B),
      t_critical =
        stats::qt(
          1 - (1 - level) / 2,
          df = pmax(B - 1L, 1L)
        ),
      halfwidth_conditional_risk =
        t_critical * mcse_conditional_risk,
      halfwidth_auc =
        t_critical * mcse_auc,
      conditional_risk_precision_met =
        is.finite(halfwidth_conditional_risk) &
        halfwidth_conditional_risk <=
          as.numeric(seq_spec$conditional_risk_halfwidth),
      auc_precision_met =
        is.finite(halfwidth_auc) &
        halfwidth_auc <=
          as.numeric(seq_spec$auc_halfwidth),
      precision_met =
        rep_sequence_complete &
        B >= as.integer(seq_spec$B_min) &
        finite_conditional_risk == B &
        finite_auc == B &
        conditional_risk_precision_met &
        auc_precision_met,
      status = dplyr::case_when(
        !rep_sequence_complete ~ "incomplete_sequence",
        B >= as.integer(seq_spec$B_max) ~ "stop_max",
        precision_met ~ "stop_precision",
        TRUE ~ "continue"
      )
    ) |>
    dplyr::arrange(
      scenario, p_over_n, depth
    )
}

make_initial_sequential_design <- function(
    spec = final_simulation_spec,
    seq_spec = sequential_mc_spec
) {
  validate_sequential_mc_spec(seq_spec)
  validate_frozen_neural_grid(spec)

  if (as.integer(spec$B) < as.integer(seq_spec$B_max)) {
    stop("final_simulation_spec$B must be at least sequential B_max.", call. = FALSE)
  }

  cells <- make_production_design(spec)$neural
  B_initial <- as.integer(seq_spec$B_min)
  batch <- as.integer(seq_spec$batch_reps)
  chunks <- as.integer(ceiling(B_initial / batch))

  tidyr::crossing(
    cell_id = cells$cell_id,
    chunk_id = seq_len(chunks)
  ) |>
    dplyr::left_join(cells, by = "cell_id") |>
    dplyr::mutate(
      rep_start = (chunk_id - 1L) * batch + 1L,
      rep_end = pmin(chunk_id * batch, B_initial),
      stage = 1L,
      task_id = dplyr::row_number()
    ) |>
    dplyr::select(
      task_id, stage, cell_id, chunk_id,
      scenario, n, p, p_over_n, depth,
      rep_start, rep_end, dplyr::everything()
    )
}

make_continuation_sequential_design <- function(
    precision_table,
    spec = final_simulation_spec,
    seq_spec = sequential_mc_spec,
    stage = NULL
) {
  validate_sequential_mc_spec(seq_spec)

  if (nrow(precision_table) == 0L) {
    return(tibble::tibble())
  }

  continuing <-
    precision_table |>
    dplyr::filter(status == "continue")

  if (nrow(continuing) == 0L) {
    return(tibble::tibble())
  }

  batch <- as.integer(seq_spec$batch_reps)
  B_max <- as.integer(seq_spec$B_max)

  if (is.null(stage)) {
    stage <-
      2L +
      as.integer(
        floor(
          (max(continuing$B) - as.integer(seq_spec$B_min)) /
          batch
        )
      )
  }

  continuing |>
    dplyr::transmute(
      scenario, n, p, p_over_n, depth,
      rep_start = max_rep + 1L,
      rep_end = pmin(max_rep + batch, B_max),
      stage = as.integer(stage)
    ) |>
    dplyr::filter(rep_start <= rep_end) |>
    dplyr::mutate(
      task_id = dplyr::row_number()
    ) |>
    dplyr::select(
      task_id, stage,
      scenario, n, p, p_over_n, depth,
      rep_start, rep_end
    )
}

save_sequential_stage_design <- function(
    design,
    file
) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(design, file = file, compress = FALSE)
  invisible(normalizePath(file, mustWork = FALSE))
}

run_neural_sequential_task <- function(
    task_id,
    task_design,
    results_dir = "results_sequential",
    overwrite = FALSE,
    spec = final_simulation_spec,
    verbose = TRUE
) {
  validate_frozen_neural_grid(spec)
  if (is.character(task_design) && length(task_design) == 1L) {
    task_design <- readRDS(task_design)
  }

  row <-
    task_design[
      task_design$task_id == as.integer(task_id),
      ,
      drop = FALSE
    ]

  if (nrow(row) != 1L) {
    stop("task_id must identify exactly one sequential neural task.", call. = FALSE)
  }

  rep_ids <- seq.int(
    from = as.integer(row$rep_start),
    to = as.integer(row$rep_end)
  )

  base_stub <- make_cell_stub(
    layer = "neural",
    scenario = row$scenario,
    p_over_n = row$p_over_n,
    depth = row$depth
  )

  rep_text <-
    if (length(rep_ids) == 1L) {
      sprintf("%02d", rep_ids[1L])
    } else {
      paste0(sprintf("%02d", min(rep_ids)), "_", sprintf("%02d", max(rep_ids)))
    }

  checkpoint_file <- file.path(
    results_dir,
    paste0(base_stub, "__rep_", rep_text, ".rds")
  )

  if (file.exists(checkpoint_file) && !overwrite) {
    message("Skipping completed sequential task: ", task_id)
    return(invisible(NULL))
  }

  result <- run_neural_mc_cell(
    scenario = row$scenario,
    n = as.integer(row$n),
    p_over_n = as.numeric(row$p_over_n),
    depth = as.integer(row$depth),
    B = as.integer(spec$B),
    rep_ids = rep_ids,
    n_test = as.integer(spec$n_test),
    master_seed = as.integer(spec$master_seed),
    spec = spec,
    verbose = verbose
  )

  save_neural_replication(
    result = result,
    scenario = row$scenario,
    p_over_n = row$p_over_n,
    depth = row$depth,
    rep_ids = rep_ids,
    results_dir = results_dir
  )

  invisible(result)
}

validate_compiled_neural_grid <- function(
    tuning,
    spec = final_simulation_spec
) {
  validate_frozen_neural_grid(spec)

  required <- c(
    "scenario", "n", "p", "p_over_n", "depth", "rep",
    "alpha", "lambda_nn"
  )
  missing <- setdiff(required, names(tuning))
  if (length(missing) > 0L) {
    stop(
      "Compiled tuning output is missing: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  expected_alpha <- length(spec$alpha_grid)
  expected_lambda <- length(spec$lambda_nn_grid)
  expected_candidates <- expected_alpha * expected_lambda

  grid_check <-
    tuning |>
    dplyr::group_by(
      scenario, n, p, p_over_n, depth, rep
    ) |>
    dplyr::summarise(
      n_alpha = dplyr::n_distinct(alpha),
      n_lambda = dplyr::n_distinct(lambda_nn),
      n_candidates = dplyr::n(),
      .groups = "drop"
    )

  bad <-
    grid_check |>
    dplyr::filter(
      n_alpha != expected_alpha |
      n_lambda != expected_lambda |
      n_candidates != expected_candidates
    )

  if (nrow(bad) > 0L) {
    stop(
      paste0(
        "Sequential checkpoint grid mismatch detected. Expected ",
        expected_alpha, " alpha x ", expected_lambda,
        " lambda = ", expected_candidates,
        " candidates for every replication. Do not mix stale and current checkpoints."
      ),
      call. = FALSE
    )
  }

  invisible(grid_check)
}


assess_sequential_results <- function(
    results_dir = "results_sequential",
    seq_spec = sequential_mc_spec,
    write_output = TRUE
) {
  compiled <- compile_fast_neural_results(
    results_dir = results_dir,
    write_output = FALSE
  )

  validate_compiled_neural_grid(
    compiled$tuning,
    spec = final_simulation_spec
  )

  precision <- summarise_sequential_neural_precision(
    compiled$neural_contrasts,
    seq_spec = seq_spec
  )

  if (write_output) {
    out_dir <- file.path(results_dir, "sequential")
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    write.csv(
      precision,
      file.path(out_dir, "sequential_precision_status.csv"),
      row.names = FALSE
    )
  }

  precision
}

sequential_mc_run_info <- function(
    spec = final_simulation_spec,
    seq_spec = sequential_mc_spec
) {
  validate_sequential_mc_spec(seq_spec)
  validate_frozen_neural_grid(spec)
  cells <- nrow(make_production_design(spec)$neural)
  initial_reps <- cells * as.integer(seq_spec$B_min)
  maximum_reps <- cells * as.integer(seq_spec$B_max)
  initial_tasks <- nrow(make_initial_sequential_design(spec, seq_spec))

  tibble::tibble(
    neural_cells = cells,
    B_min = as.integer(seq_spec$B_min),
    B_max = as.integer(seq_spec$B_max),
    batch_reps = as.integer(seq_spec$batch_reps),
    initial_replications = initial_reps,
    maximum_replications = maximum_reps,
    initial_tasks = initial_tasks,
    conditional_risk_halfwidth_target =
      as.numeric(seq_spec$conditional_risk_halfwidth),
    auc_halfwidth_target =
      as.numeric(seq_spec$auc_halfwidth),
    confidence_level =
      as.numeric(seq_spec$confidence_level)
  )
}
