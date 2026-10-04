# ═══════════════════════════════════════════════════════════════════════
# Essay 3 — Insider Trading: Fundamental vs Cultural
# R translation of model/essay3.py
#
# Core pipeline: build_insider_panel() + run_essay3()
#   - Abnormal net trading panel
#   - TOST equivalence (TOSTER)
#   - Wilcoxon distributional tests
#   - Informed trading directional accuracy
#   - Proximity gradients
#   - Fama-MacBeth cross-sectional regressions
#   - Wild cluster bootstrap (fwildclusterboot)
#   - Placebo permutation tests
#
# Required packages:
#   tidyverse, fixest, TOSTER, fwildclusterboot, sandwich, lmtest,
#   broom, zoo, quantreg
# ═══════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(tidyverse)
  library(fixest)
  library(TOSTER)
  library(fwildclusterboot)
  library(sandwich)
  library(lmtest)
  library(broom)
  library(zoo)
})

# ── Constants ─────────────────────────────────────────────────────────

WINDOWS <- list(
  BENCHMARK  = c(-365, -181),
  PRE_FAR    = c(-180, -61),
  PRE_MID    = c(-60,  -31),
  PRE_NEAR   = c(-30,   -1),
  PRE_FULL   = c(-180,  -1),
  POST       = c(0,     60),
  POST_EARLY = c(0,     10)
)

SESOI_D <- 0.20

REGULATORY_PERIODS <- tibble(
  period = c("PRE_SOX", "SOX_ERA", "DODD_FRANK", "POST_AMENDMENTS"),
  start  = as.Date(c("2000-01-01", "2003-01-01", "2010-01-01", "2023-01-01")),
  end    = as.Date(c("2002-12-31", "2009-12-31", "2022-12-31", "2030-12-31"))
)


# ── Helpers ───────────────────────────────────────────────────────────

assign_regulatory_period <- function(event_date) {
  event_date <- as.Date(event_date)
  for (i in seq_len(nrow(REGULATORY_PERIODS))) {
    if (event_date >= REGULATORY_PERIODS$start[i] &&
        event_date <= REGULATORY_PERIODS$end[i]) {
      return(REGULATORY_PERIODS$period[i])
    }
  }
  return("PRE_SOX")
}

gini_coefficient <- function(values) {
  # Gini for non-negative values
  values <- sort(abs(values))
  n <- length(values)
  if (n == 0 || sum(values) == 0) return(0.0)
  idx <- seq_len(n)
  (2 * sum(idx * values)) / (n * sum(values)) - (n + 1) / n
}

benjamini_hochberg <- function(pvals, alpha = 0.05) {
  # BH step-up procedure. Returns logical vector of significance.
  n <- length(pvals)
  if (n == 0) return(logical(0))
  ord <- order(pvals)
  sorted_p <- pvals[ord]
  thresholds <- (seq_len(n) / n) * alpha
  # Find largest k where p_(k) <= k/n * alpha
  k_max <- max(c(0, which(sorted_p <= thresholds)))
  significant <- logical(n)
  if (k_max > 0) significant[ord[1:k_max]] <- TRUE
  significant
}


# ═══════════════════════════════════════════════════════════════════════
# TOST EQUIVALENCE
# ═══════════════════════════════════════════════════════════════════════

compute_tost <- function(vals, sesoi_d = SESOI_D) {
  #' TOST equivalence test against zero.
  #'

  #' @param vals Numeric vector of observations


#' @param sesoi_d Smallest effect size of interest (Cohen's d)
  #' @return Named list with TOST results
  vals <- vals[!is.na(vals)]
  n <- length(vals)
  if (n < 5) return(NULL)

  mean_val <- mean(vals)
  std_val  <- sd(vals)
  se       <- std_val / sqrt(n)
  delta    <- sesoi_d * std_val

  if (se > 0) {
    t_upper <- (mean_val - delta) / se
    t_lower <- (mean_val + delta) / se
    p_upper <- pt(t_upper, df = n - 1)
    p_lower <- 1 - pt(t_lower, df = n - 1)
  } else {
    t_upper <- NA_real_
    t_lower <- NA_real_
    p_upper <- NA_real_
    p_lower <- NA_real_
  }
  p_tost <- max(p_upper, p_lower, na.rm = TRUE)

  # Power at observed effect
  ncp <- if (se > 0) mean_val / se else 0
  power <- if (se > 0) {
    1 - pt(qt(0.975, df = n - 1), df = n - 1, ncp = abs(ncp))
  } else NA_real_

  # Minimum detectable effect (80% power)
  z_alpha <- qnorm(0.975)
  z_beta  <- qnorm(0.80)
  mde_raw <- if (se > 0) (z_alpha + z_beta) * se else NA_real_
  mde_d   <- if (std_val > 0 && !is.na(mde_raw)) mde_raw / std_val else NA_real_

  list(
    N = n, MEAN = mean_val, STD = std_val, SE = se,
    SESOI_D = sesoi_d, DELTA_RAW = delta,
    P_UPPER = p_upper, P_LOWER = p_lower, P_TOST = p_tost,
    EQUIVALENT = p_tost < 0.05,
    POWER_AT_OBSERVED = power, MDE_80_D = mde_d, MDE_80_RAW = mde_raw
  )
}

compute_tost_equivalence <- function(panel) {
  #' TOST equivalence tests by event category.
  #'
  #' @param panel data.frame with HAS_SUFFICIENT_DATA, EVENT_CATEGORY,
  #'   ABNORMAL_NET_TRADING columns
  #' @return data.frame of TOST results
  if (nrow(panel) == 0) return(data.frame())

  df <- panel %>% filter(HAS_SUFFICIENT_DATA == 1)
  results <- list()

  for (cat in c("CULTURAL", "FUNDAMENTAL")) {
    vals <- df %>% filter(EVENT_CATEGORY == cat) %>%
      pull(ABNORMAL_NET_TRADING) %>% na.omit()
    if (length(vals) >= 10) {
      res <- compute_tost(vals, SESOI_D)
      res$EVENT_CATEGORY <- cat
      res$TEST <- "TOST"
      results <- c(results, list(as_tibble(res)))
    }
  }

  # Matched fundamental
  if ("MATCHED" %in% names(df)) {
    vals <- df %>%
      filter(EVENT_CATEGORY == "FUNDAMENTAL", MATCHED == TRUE) %>%
      pull(ABNORMAL_NET_TRADING) %>% na.omit()
    if (length(vals) >= 10) {
      res <- compute_tost(vals, SESOI_D)
      res$EVENT_CATEGORY <- "FUNDAMENTAL_MATCHED"
      res$TEST <- "TOST"
      results <- c(results, list(as_tibble(res)))
    }
  }

  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}


# ═══════════════════════════════════════════════════════════════════════
# MEAN vs DISTRIBUTIONAL (Wilcoxon, t-test, effect sizes)
# ═══════════════════════════════════════════════════════════════════════

compute_mean_vs_distributional <- function(panel) {
  #' Core mean-vs-distributional analysis for insider trading.
  #'
  #' Tests whether the distributional shift (Wilcoxon) is significant
  #' even when the mean (t-test) is not.
  #'
  #' @param panel data.frame with HAS_SUFFICIENT_DATA, EVENT_CATEGORY,
  #'   ABNORMAL_NET_TRADING, EVENT_TYPE, HIGH_POLITICAL_CONNECTION,
  #'   HIGH_OPP, CONDITIONS_MET columns
  #' @return data.frame of test results
  if (nrow(panel) == 0) return(data.frame())

  df <- panel %>% filter(HAS_SUFFICIENT_DATA == 1)
  dv <- "ABNORMAL_NET_TRADING"
  results <- list()

  .test_subset <- function(vals, cut_label, subset_label) {
    vals <- vals[!is.na(vals)]
    n <- length(vals)
    if (n < 10) return(NULL)

    t_res <- t.test(vals, mu = 0)
    w_res <- tryCatch(
      wilcox.test(vals, mu = 0, exact = FALSE),
      error = function(e) list(statistic = NA, p.value = NA)
    )
    d <- mean(vals) / sd(vals)

    tibble(
      CUT = cut_label, SUBSET = subset_label, N = n,
      MEAN = mean(vals), MEDIAN = median(vals), STD = sd(vals),
      T_STAT = t_res$statistic, T_PVALUE = t_res$p.value,
      WILCOXON_STAT = as.numeric(w_res$statistic),
      WILCOXON_PVALUE = w_res$p.value,
      COHEN_D = d,
      PCT_POSITIVE = mean(vals > 0)
    )
  }

  # Overall by category
  for (cat in c("FUNDAMENTAL", "CULTURAL")) {
    vals <- df %>% filter(EVENT_CATEGORY == cat) %>% pull(!!sym(dv))
    r <- .test_subset(vals, cat, "ALL")
    if (!is.null(r)) results <- c(results, list(r))
  }

  # By event type within fundamental
  for (etype in unique(df$EVENT_TYPE[df$EVENT_CATEGORY == "FUNDAMENTAL"])) {
    vals <- df %>%
      filter(EVENT_CATEGORY == "FUNDAMENTAL", EVENT_TYPE == etype) %>%
      pull(!!sym(dv))
    r <- .test_subset(vals, "FUNDAMENTAL", etype)
    if (!is.null(r)) results <- c(results, list(r))
  }

  # High political connection
  if ("HIGH_POLITICAL_CONNECTION" %in% names(df)) {
    vals <- df %>%
      filter(EVENT_CATEGORY == "FUNDAMENTAL",
             HIGH_POLITICAL_CONNECTION == TRUE) %>%
      pull(!!sym(dv))
    r <- .test_subset(vals, "FUNDAMENTAL", "HIGH_POLITICAL_CONNECTION")
    if (!is.null(r)) results <- c(results, list(r))
  }

  # High opportunistic
  if ("HIGH_OPP" %in% names(df)) {
    vals <- df %>%
      filter(EVENT_CATEGORY == "FUNDAMENTAL", HIGH_OPP == TRUE) %>%
      pull(!!sym(dv))
    r <- .test_subset(vals, "FUNDAMENTAL", "HIGH_OPP")
    if (!is.null(r)) results <- c(results, list(r))
  }

  # Conditions met == 4
  if ("CONDITIONS_MET" %in% names(df)) {
    vals <- df %>%
      filter(EVENT_CATEGORY == "FUNDAMENTAL", CONDITIONS_MET == 4) %>%
      pull(!!sym(dv))
    r <- .test_subset(vals, "FUNDAMENTAL", "CONDITIONS_MET_4")
    if (!is.null(r)) results <- c(results, list(r))
  }

  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}


# ═══════════════════════════════════════════════════════════════════════
# WILCOXON FAMILY CORRECTION (Gating Test)
# ═══════════════════════════════════════════════════════════════════════

compute_wilcoxon_family <- function(mean_vs_dist) {
  #' BH and Holm correction across the Wilcoxon test family.
  #'
  #' @param mean_vs_dist data.frame from compute_mean_vs_distributional()
  #' @return data.frame with BH_SIGNIFICANT and HOLM_SIGNIFICANT columns
  if (nrow(mean_vs_dist) == 0) return(data.frame())

  df <- mean_vs_dist %>%
    filter(!is.na(WILCOXON_PVALUE))

  if (nrow(df) == 0) return(data.frame())

  df$BH_ADJUSTED   <- p.adjust(df$WILCOXON_PVALUE, method = "BH")
  df$HOLM_ADJUSTED  <- p.adjust(df$WILCOXON_PVALUE, method = "holm")
  df$BH_SIGNIFICANT   <- as.integer(df$BH_ADJUSTED < 0.05)
  df$HOLM_SIGNIFICANT <- as.integer(df$HOLM_ADJUSTED < 0.05)
  # Divergence: Wilcoxon significant but t-test not
  df$DIVERGENCE <- ifelse(
    df$WILCOXON_PVALUE < 0.05 & df$T_PVALUE >= 0.05,
    "YES", "NO"
  )

  df
}


# ═══════════════════════════════════════════════════════════════════════
# BOOTSTRAP WILCOXON (sign-flip)
# ═══════════════════════════════════════════════════════════════════════

compute_bootstrap_wilcoxon <- function(panel, n_bootstrap = 1000, seed = 42) {
  #' Bootstrap the Wilcoxon test via sign-flipping.
  #'
  #' @param panel data.frame
  #' @param n_bootstrap integer
  #' @param seed integer
  #' @return data.frame
  if (nrow(panel) == 0) return(data.frame())

  df <- panel %>% filter(HAS_SUFFICIENT_DATA == 1)
  dv <- "ABNORMAL_NET_TRADING"
  set.seed(seed)
  results <- list()

  cuts <- list(
    FUNDAMENTAL_ALL = df %>%
      filter(EVENT_CATEGORY == "FUNDAMENTAL") %>%
      pull(!!sym(dv)) %>% na.omit()
  )
  if ("EVENT_TYPE" %in% names(df)) {
    cuts$COURT_DECISION <- df %>%
      filter(EVENT_CATEGORY == "FUNDAMENTAL",
             EVENT_TYPE == "COURT_DECISION") %>%
      pull(!!sym(dv)) %>% na.omit()
  }

  for (label in names(cuts)) {
    vals <- cuts[[label]]
    if (length(vals) < 10) next

    obs_w <- tryCatch(
      wilcox.test(vals, mu = 0, exact = FALSE),
      error = function(e) list(statistic = NA, p.value = NA)
    )

    abs_vals <- abs(vals)
    boot_pvals <- replicate(n_bootstrap, {
      signs <- sample(c(-1, 1), length(abs_vals), replace = TRUE)
      samp <- signs * abs_vals
      tryCatch(
        wilcox.test(samp, mu = 0, exact = FALSE)$p.value,
        error = function(e) NA_real_
      )
    })
    boot_pvals <- boot_pvals[!is.na(boot_pvals)]

    if (length(boot_pvals) == 0) next

    results <- c(results, list(tibble(
      CUT = label, N = length(vals),
      OBSERVED_STAT = as.numeric(obs_w$statistic),
      OBSERVED_PVALUE = obs_w$p.value,
      BOOT_PVALUE_MEAN = mean(boot_pvals),
      BOOT_PVALUE_MEDIAN = median(boot_pvals),
      BOOT_PCT_SIG_005 = mean(boot_pvals < 0.05),
      BOOT_PCT_SIG_010 = mean(boot_pvals < 0.10),
      N_BOOTSTRAP = n_bootstrap
    )))
  }

  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}


# ═══════════════════════════════════════════════════════════════════════
# TRIMMED ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════

compute_trimmed_robustness <- function(panel) {
  #' Re-run core tests after trimming top/bottom 1% and 5%.
  #'
  #' @param panel data.frame
  #' @return data.frame
  if (nrow(panel) == 0) return(data.frame())

  df <- panel %>% filter(HAS_SUFFICIENT_DATA == 1)
  dv <- "ABNORMAL_NET_TRADING"
  results <- list()

  for (trim_pct in c(1, 5)) {
    fund_vals <- df %>%
      filter(EVENT_CATEGORY == "FUNDAMENTAL") %>%
      pull(!!sym(dv)) %>% na.omit()
    lo <- quantile(fund_vals, trim_pct / 100)
    hi <- quantile(fund_vals, 1 - trim_pct / 100)
    trimmed <- df %>% filter(!!sym(dv) >= lo, !!sym(dv) <= hi)

    .test_trimmed <- function(sub_vals, cut_label) {
      sub_vals <- sub_vals[!is.na(sub_vals)]
      if (length(sub_vals) < 10) return(NULL)
      t_res <- t.test(sub_vals, mu = 0)
      w_res <- tryCatch(
        wilcox.test(sub_vals, mu = 0, exact = FALSE),
        error = function(e) list(statistic = NA, p.value = NA)
      )
      tibble(
        TRIM_PCT = trim_pct, CUT = cut_label,
        N = length(sub_vals),
        MEAN = mean(sub_vals), MEDIAN = median(sub_vals),
        T_STAT = t_res$statistic, T_PVALUE = t_res$p.value,
        WILCOXON_STAT = as.numeric(w_res$statistic),
        WILCOXON_PVALUE = w_res$p.value,
        PCT_POSITIVE = mean(sub_vals > 0)
      )
    }

    r <- .test_trimmed(
      trimmed %>% filter(EVENT_CATEGORY == "FUNDAMENTAL") %>%
        pull(!!sym(dv)),
      "FUNDAMENTAL_ALL"
    )
    if (!is.null(r)) results <- c(results, list(r))

    if ("EVENT_TYPE" %in% names(trimmed)) {
      r <- .test_trimmed(
        trimmed %>%
          filter(EVENT_CATEGORY == "FUNDAMENTAL",
                 EVENT_TYPE == "COURT_DECISION") %>%
          pull(!!sym(dv)),
        "COURT_DECISION"
      )
      if (!is.null(r)) results <- c(results, list(r))
    }
  }

  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}


# ═══════════════════════════════════════════════════════════════════════
# PLACEBO PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════

compute_placebo_test <- function(panel, n_iterations = 500, seed = 42) {
  #' Permutation test: shuffle event category labels.
  #'
  #' @param panel data.frame
  #' @param n_iterations integer
  #' @param seed integer
  #' @return data.frame with placebo test results
  if (nrow(panel) == 0) return(data.frame())

  df <- panel %>% filter(HAS_SUFFICIENT_DATA == 1)
  dv <- "ABNORMAL_NET_TRADING"

  fund <- df %>% filter(EVENT_CATEGORY == "FUNDAMENTAL") %>%
    pull(!!sym(dv)) %>% na.omit()
  cult <- df %>% filter(EVENT_CATEGORY == "CULTURAL") %>%
    pull(!!sym(dv)) %>% na.omit()

  if (length(fund) < 5 || length(cult) < 5) return(data.frame())

  observed <- mean(fund) - mean(cult)
  all_vals <- c(fund, cult)
  n_fund <- length(fund)
  set.seed(seed)

  null_diffs <- replicate(n_iterations, {
    s <- sample(all_vals)
    mean(s[1:n_fund]) - mean(s[(n_fund + 1):length(s)])
  })

  p_value <- mean(abs(null_diffs) >= abs(observed))

  tibble(
    TEST = "PLACEBO_PERMUTATION",
    OBSERVED_DIFF = observed,
    NULL_MEAN = mean(null_diffs),
    NULL_STD = sd(null_diffs),
    P_VALUE = p_value,
    N_ITERATIONS = n_iterations,
    CI_2_5 = quantile(null_diffs, 0.025),
    CI_97_5 = quantile(null_diffs, 0.975)
  )
}


# ═══════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════

compute_bootstrap_ci <- function(panel, n_bootstrap = 1000, seed = 42) {
  #' Bootstrap CI for the fundamental-cultural contrast.
  #'
  #' @param panel data.frame
  #' @param n_bootstrap integer
  #' @param seed integer
  #' @return data.frame
  if (nrow(panel) == 0) return(data.frame())

  df <- panel %>% filter(HAS_SUFFICIENT_DATA == 1)
  dv <- "ABNORMAL_NET_TRADING"

  fund <- df %>% filter(EVENT_CATEGORY == "FUNDAMENTAL") %>%
    pull(!!sym(dv)) %>% na.omit()
  cult <- df %>% filter(EVENT_CATEGORY == "CULTURAL") %>%
    pull(!!sym(dv)) %>% na.omit()

  if (length(fund) < 5 || length(cult) < 5) return(data.frame())

  observed <- mean(fund) - mean(cult)
  set.seed(seed)

  boot_diffs <- replicate(n_bootstrap, {
    mean(sample(fund, length(fund), replace = TRUE)) -
      mean(sample(cult, length(cult), replace = TRUE))
  })

  tibble(
    TEST = "BOOTSTRAP_CI",
    OBSERVED_DIFF = observed,
    BOOT_MEAN = mean(boot_diffs),
    BOOT_STD = sd(boot_diffs),
    CI_2_5 = quantile(boot_diffs, 0.025),
    CI_97_5 = quantile(boot_diffs, 0.975),
    CI_5 = quantile(boot_diffs, 0.05),
    CI_95 = quantile(boot_diffs, 0.95),
    N_BOOTSTRAP = n_bootstrap,
    ZERO_IN_95CI = as.integer(
      quantile(boot_diffs, 0.025) <= 0 &&
        0 <= quantile(boot_diffs, 0.975)
    )
  )
}


# ═══════════════════════════════════════════════════════════════════════
# INSIDER CONCENTRATION (Gini, HHI, Top-K%)
# ═══════════════════════════════════════════════════════════════════════

compute_insider_concentration <- function(insider_profits) {
  #' Concentration metrics by event category.
  #'
  #' @param insider_profits data.frame with EVENT_CATEGORY, ABNORMAL_NET_SELLING
  #' @return data.frame
  if (nrow(insider_profits) == 0) return(data.frame())

  results <- list()
  for (cat in unique(insider_profits$EVENT_CATEGORY)) {
    sub <- insider_profits %>%
      filter(EVENT_CATEGORY == cat, !is.na(ABNORMAL_NET_SELLING))
    abs_profits <- abs(sub$ABNORMAL_NET_SELLING)

    if (length(abs_profits) < 5) next
    total_abs <- sum(abs_profits)
    if (total_abs == 0) next

    n <- length(abs_profits)
    sorted_abs <- sort(abs_profits, decreasing = TRUE)

    # Top-K% shares
    for (k_pct in c(1, 5, 10, 20, 50)) {
      k <- max(1, ceiling(n * k_pct / 100))
      share <- sum(sorted_abs[1:k]) / total_abs
      results <- c(results, list(tibble(
        EVENT_CATEGORY = cat, METRIC = paste0("TOP_", k_pct, "PCT_SHARE"),
        VALUE = share, K = k, N_INSIDERS = n,
        TOTAL_ABS_NET_SELLING = total_abs
      )))
    }

    # Gini
    results <- c(results, list(tibble(
      EVENT_CATEGORY = cat, METRIC = "GINI",
      VALUE = gini_coefficient(abs_profits), K = NA_real_,
      N_INSIDERS = n, TOTAL_ABS_NET_SELLING = total_abs
    )))

    # HHI
    shares <- abs_profits / total_abs
    results <- c(results, list(tibble(
      EVENT_CATEGORY = cat, METRIC = "HHI",
      VALUE = sum(shares^2), K = NA_real_,
      N_INSIDERS = n, TOTAL_ABS_NET_SELLING = total_abs
    )))
  }

  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}


# ═══════════════════════════════════════════════════════════════════════
# INSIDER FIXED EFFECTS (within-insider variation)
# ═══════════════════════════════════════════════════════════════════════

compute_insider_fe <- function(insider_profits) {
  #' Within-insider variation via fixest::feols().
  #'
  #' @param insider_profits data.frame with OWNER, EVENT_DATE,
  #'   ABNORMAL_NET_SELLING, EVENT_CATEGORY, EVENT_TYPE
  #' @return data.frame of regression results
  if (nrow(insider_profits) < 30) return(data.frame())

  ip <- insider_profits %>%
    mutate(
      IS_FUNDAMENTAL = as.integer(EVENT_CATEGORY == "FUNDAMENTAL"),
      IS_COURT = as.integer(EVENT_TYPE == "COURT_DECISION")
    )

  # Keep only repeat insiders (>= 2 events)
  repeat_owners <- ip %>%
    count(OWNER) %>%
    filter(n >= 2) %>%
    pull(OWNER)

  ip_repeat <- ip %>% filter(OWNER %in% repeat_owners)
  if (nrow(ip_repeat) < 20 || n_distinct(ip_repeat$OWNER) < 10) {
    return(data.frame())
  }

  results <- list()

  # Insider FE with fixest
  tryCatch({
    fe_model <- feols(
      ABNORMAL_NET_SELLING ~ IS_FUNDAMENTAL + IS_COURT | OWNER,
      data = ip_repeat,
      cluster = ~OWNER
    )
    s <- summary(fe_model)
    ct <- as.data.frame(s$coeftable)
    ct$VARIABLE <- rownames(ct)
    for (i in seq_len(nrow(ct))) {
      results <- c(results, list(tibble(
        SPECIFICATION = "INSIDER_FE",
        VARIABLE = ct$VARIABLE[i],
        COEFFICIENT = ct$Estimate[i],
        STD_ERROR = ct$`Std. Error`[i],
        T_STAT = ct$`t value`[i],
        P_VALUE = ct$`Pr(>|t|)`[i],
        N_OBS = nobs(fe_model),
        N_INSIDERS = n_distinct(ip_repeat$OWNER),
        R_SQUARED = r2(fe_model, type = "within")
      )))
    }
  }, error = function(e) {
    message("Insider FE failed: ", e$message)
  })

  # Pooled OLS for comparison
  tryCatch({
    if ("HIGH_POLITICAL_CONNECTION" %in% names(ip) &&
        "HIGH_OPP" %in% names(ip)) {
      ip$HIGH_CONN_INT <- as.integer(ip$HIGH_POLITICAL_CONNECTION)
      ip$HIGH_OPP_INT  <- as.integer(ip$HIGH_OPP)
      pooled <- feols(
        ABNORMAL_NET_SELLING ~ IS_FUNDAMENTAL + IS_COURT +
          HIGH_CONN_INT + HIGH_OPP_INT,
        data = ip, cluster = ~OWNER
      )
      s <- summary(pooled)
      ct <- as.data.frame(s$coeftable)
      ct$VARIABLE <- rownames(ct)
      for (i in seq_len(nrow(ct))) {
        results <- c(results, list(tibble(
          SPECIFICATION = "POOLED_OLS",
          VARIABLE = ct$VARIABLE[i],
          COEFFICIENT = ct$Estimate[i],
          STD_ERROR = ct$`Std. Error`[i],
          T_STAT = ct$`t value`[i],
          P_VALUE = ct$`Pr(>|t|)`[i],
          N_OBS = nobs(pooled),
          N_INSIDERS = n_distinct(ip$OWNER),
          R_SQUARED = r2(pooled)
        )))
      }
    }
  }, error = function(e) {
    message("Pooled OLS failed: ", e$message)
  })

  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}


# ═══════════════════════════════════════════════════════════════════════
# INFORMED TRADING DIRECTIONAL ACCURACY
# ═══════════════════════════════════════════════════════════════════════

conditional_sign_base_rate <- function(event_cars) {
  #' Pesaran-Timmermann (1992) conditional base rate.
  #'
  #' Fraction of events with CAR < 0.
  #' @param event_cars numeric vector of CARs
  #' @return numeric
  cars <- event_cars[!is.na(event_cars)]
  if (length(cars) == 0) return(0.5)
  mean(cars < 0)
}

compute_informed_trading <- function(trades, panel) {
  #' Headline informed trading directional accuracy test.
  #'
  #' Uses event-window CAR to define event-profitability:
  #'   - Sell profitable if CAR_POST < 0
  #'   - Buy profitable if CAR_POST > 0
  #'
  #' @param trades data.frame with TRADE_TYPE, EVENT_CAR,
  #'   EVENT_PROFITABLE, TRADE_VALUE, EVENT_ID, etc.
  #' @param panel data.frame with EVENT_ID, CAR_POST
  #' @return named list with informed_trading, informed_proximity
  if (nrow(trades) == 0) {
    return(list(informed_trading = data.frame(),
                informed_proximity = data.frame()))
  }

  valid_pol <- trades %>% filter(!is.na(EVENT_PROFITABLE))
  if (nrow(valid_pol) < 10) {
    return(list(informed_trading = data.frame(),
                informed_proximity = data.frame()))
  }

  # Conditional base rates
  analysis_cars <- valid_pol %>%
    group_by(EVENT_ID) %>%
    summarise(EVENT_CAR = first(EVENT_CAR), .groups = "drop") %>%
    pull(EVENT_CAR) %>% na.omit()
  p_cond_sell <- conditional_sign_base_rate(analysis_cars)
  p_cond_buy  <- 1 - p_cond_sell

  .summarize <- function(sub, label, p_cond = NULL) {
    n <- nrow(sub)
    if (n < 5) return(NULL)
    pct <- mean(sub$EVENT_PROFITABLE)
    n_prof <- sum(sub$EVENT_PROFITABLE)
    binom_50 <- binom.test(n_prof, n, p = 0.5)$p.value
    binom_cond <- if (!is.null(p_cond) && p_cond > 0 && p_cond < 1) {
      binom.test(n_prof, n, p = p_cond)$p.value
    } else NA_real_

    tibble(
      CUT = label, N_TRADES = n,
      METRIC_VALUE = pct,
      METRIC_PVAL = binom_50,
      METRIC_PVAL_COND = binom_cond,
      COND_NULL = if (!is.null(p_cond)) p_cond else NA_real_,
      MEAN_TRADE_VALUE = mean(sub$TRADE_VALUE, na.rm = TRUE),
      MEAN_EVENT_CAR = mean(sub$EVENT_CAR, na.rm = TRUE)
    )
  }

  summary_rows <- list()
  r <- .summarize(valid_pol, "ALL")
  if (!is.null(r)) summary_rows <- c(summary_rows, list(r))

  sells <- valid_pol %>% filter(TRADE_TYPE == "sell")
  buys  <- valid_pol %>% filter(TRADE_TYPE == "buy")
  r <- .summarize(sells, "SELLS_ONLY", p_cond_sell)
  if (!is.null(r)) summary_rows <- c(summary_rows, list(r))
  r <- .summarize(buys, "BUYS_ONLY", p_cond_buy)
  if (!is.null(r)) summary_rows <- c(summary_rows, list(r))

  informed_trading <- if (length(summary_rows) > 0) {
    bind_rows(summary_rows)
  } else data.frame()

  # Proximity windows
  proximity_rows <- list()
  windows <- list(c(0, 30), c(31, 60), c(61, 90), c(91, 180))
  for (w in windows) {
    sub <- valid_pol %>%
      filter(DAYS_BEFORE_EVENT >= w[1], DAYS_BEFORE_EVENT <= w[2])
    label <- paste0(w[1], "-", w[2], "d")
    r <- .summarize(sub, label)
    if (!is.null(r)) {
      r$WINDOW_START <- w[1]
      r$WINDOW_END   <- w[2]
      proximity_rows <- c(proximity_rows, list(r))
    }
    # Sells only
    sub_sells <- sub %>% filter(TRADE_TYPE == "sell")
    r <- .summarize(sub_sells, paste0(label, "_SELLS"))
    if (!is.null(r)) {
      r$WINDOW_START <- w[1]
      r$WINDOW_END   <- w[2]
      proximity_rows <- c(proximity_rows, list(r))
    }
  }

  informed_proximity <- if (length(proximity_rows) > 0) {
    bind_rows(proximity_rows)
  } else data.frame()

  list(
    informed_trading = informed_trading,
    informed_proximity = informed_proximity,
    p_cond_sell = p_cond_sell,
    p_cond_buy = p_cond_buy
  )
}


# ═══════════════════════════════════════════════════════════════════════
# WILD CLUSTER BOOTSTRAP FOR PROPORTIONS
# ═══════════════════════════════════════════════════════════════════════

wild_cluster_bootstrap_proportion <- function(outcomes, clusters,
                                               p0 = 0.5,
                                               n_boot = 1000,
                                               seed = 42) {
  #' Wild cluster bootstrap CI and test for a proportion.
  #'
  #' Cameron-Gelbach-Miller (2011); MacKinnon-Webb (2017, 2018).
  #' Uses 2-point Rademacher weights unless G_eff < 12, then Webb 6-point.
  #'
  #' @param outcomes numeric 0/1 vector
  #' @param clusters cluster identifiers
  #' @param p0 null hypothesis proportion
  #' @param n_boot number of bootstrap iterations
  #' @param seed random seed
  #' @return named list or NULL if < 10 valid obs
  valid <- !is.na(outcomes)
  outcomes <- outcomes[valid]
  clusters <- clusters[valid]
  n <- length(outcomes)
  if (n < 10) return(NULL)

  cl_unique <- unique(clusters)
  G <- length(cl_unique)

  # Effective cluster count (MacKinnon-Webb 2017)
  n_g <- table(clusters)
  g_eff <- sum(n_g)^2 / sum(n_g^2)

  p_hat <- mean(outcomes)

  # Cluster-level quantities
  y_bar_g <- tapply(outcomes, clusters, mean)
  w_g     <- as.numeric(n_g) / n

  # Weight distribution
  use_webb <- g_eff < 12
  if (use_webb) {
    weight_pool <- c(-sqrt(1.5), -1, -sqrt(0.5),
                      sqrt(0.5),  1,  sqrt(1.5))
    weights_used <- "webb6"
  } else {
    weight_pool <- c(-1, 1)
    weights_used <- "rademacher2"
  }

  set.seed(seed)
  boot_props <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    eps <- sample(weight_pool, G, replace = TRUE)
    boot_props[b] <- p_hat + sum(w_g * (y_bar_g[names(n_g)] - p_hat) * eps)
  }

  cluster_se <- sd(boot_props)
  ci_lo_95 <- quantile(boot_props, 0.025)
  ci_hi_95 <- quantile(boot_props, 0.975)

  # Bootstrap test vs p0
  t_obs <- abs(p_hat - p0)
  set.seed(seed + 1)
  t_star <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    eps <- sample(weight_pool, G, replace = TRUE)
    t_star[b] <- abs(sum(w_g * (y_bar_g[names(n_g)] - p0) * eps))
  }
  p_value <- mean(t_star >= t_obs)

  list(
    point_est = p_hat,
    cluster_se = cluster_se,
    ci_lo_95 = as.numeric(ci_lo_95),
    ci_hi_95 = as.numeric(ci_hi_95),
    n_obs = n,
    g_raw = G,
    g_eff = g_eff,
    p_value_vs_p0 = p_value,
    p0 = p0,
    weights_used = weights_used
  )
}


# ═══════════════════════════════════════════════════════════════════════
# DATE-RANDOMIZATION PLACEBO (directional accuracy)
# ═══════════════════════════════════════════════════════════════════════

placebo_directional <- function(trades, n_perms = 1000, seed = 42) {
  #' Date-randomization placebo for directional accuracy.
  #'
  #' Shuffles EVENT_CAR labels across events, recomputes accuracy.
  #'
  #' @param trades data.frame with TRADE_TYPE, EVENT_CAR, EVENT_ID
  #' @return data.frame (single row) or empty
  if (nrow(trades) == 0 || !("EVENT_CAR" %in% names(trades))) {
    return(data.frame())
  }

  valid <- trades %>%
    filter(!is.na(EVENT_CAR), TRADE_TYPE %in% c("buy", "sell"))
  if (nrow(valid) < 10) return(data.frame())

  .compute_accuracy <- function(df) {
    sells <- df %>% filter(TRADE_TYPE == "sell")
    buys  <- df %>% filter(TRADE_TYPE == "buy")
    correct <- sum(sells$EVENT_CAR < 0) + sum(buys$EVENT_CAR > 0)
    total <- nrow(sells) + nrow(buys)
    if (total == 0) return(NA_real_)
    correct / total
  }

  observed_acc <- .compute_accuracy(valid)
  if (is.na(observed_acc)) return(data.frame())

  event_car_map <- valid %>%
    group_by(EVENT_ID) %>%
    summarise(EVENT_CAR = first(EVENT_CAR), .groups = "drop")
  event_ids  <- event_car_map$EVENT_ID
  car_values <- event_car_map$EVENT_CAR

  set.seed(seed)
  perm_accs <- replicate(n_perms, {
    shuffled <- sample(car_values)
    car_lookup <- setNames(shuffled, event_ids)
    perm_df <- valid
    perm_df$EVENT_CAR <- car_lookup[as.character(perm_df$EVENT_ID)]
    .compute_accuracy(perm_df)
  })
  perm_accs <- perm_accs[!is.na(perm_accs)]
  p_value <- mean(perm_accs >= observed_acc)


  tibble(
    TEST = "DATE_RANDOMIZATION_PLACEBO_DIRECTIONAL",
    OBSERVED_ACCURACY = observed_acc,
    PERM_MEAN = mean(perm_accs),
    PERM_STD = sd(perm_accs),
    PERM_CI_2_5 = quantile(perm_accs, 0.025),
    PERM_CI_97_5 = quantile(perm_accs, 0.975),
    P_VALUE_VS_PERMUTATION = p_value,
    N_OBS = nrow(valid),
    N_EVENTS = length(event_ids),
    N_PERMS = n_perms
  )
}


# ═══════════════════════════════════════════════════════════════════════
# FAMA-MACBETH CROSS-SECTIONAL REGRESSIONS
# ═══════════════════════════════════════════════════════════════════════

fama_macbeth <- function(panel, dv = "ABNORMAL_NET_TRADING",
                          time_col = "EVENT_YEAR") {
  #' Fama-MacBeth (1973) cross-sectional regression.
  #'
  #' Run cross-section regressions period by period, then average
  #' coefficients and compute Newey-West adjusted t-stats.
  #'
  #' @param panel data.frame with dv, time_col, IS_FUNDAMENTAL, etc.
  #' @param dv dependent variable name
  #' @param time_col time period column
  #' @return data.frame of average coefficients and t-stats
  if (nrow(panel) == 0) return(data.frame())

  df <- panel %>% filter(HAS_SUFFICIENT_DATA == 1)
  if (!(time_col %in% names(df))) return(data.frame())

  # Build regressors
  df <- df %>%
    mutate(
      IS_FUNDAMENTAL = as.integer(EVENT_CATEGORY == "FUNDAMENTAL")
    )

  x_cols <- "IS_FUNDAMENTAL"
  if ("HIGH_POLITICAL_CONNECTION" %in% names(df)) {
    df$HIGH_CONN_INT <- as.integer(df$HIGH_POLITICAL_CONNECTION)
    x_cols <- c(x_cols, "HIGH_CONN_INT")
  }
  if ("HIGH_OPP" %in% names(df)) {
    df$HIGH_OPP_INT <- as.integer(df$HIGH_OPP)
    x_cols <- c(x_cols, "HIGH_OPP_INT")
  }

  periods <- sort(unique(df[[time_col]]))
  if (length(periods) < 3) return(data.frame())

  # Cross-section regression per period
  coef_list <- list()
  for (p in periods) {
    sub <- df %>% filter(!!sym(time_col) == p)
    if (nrow(sub) < 10) next

    formula_str <- paste0(dv, " ~ ", paste(x_cols, collapse = " + "))
    tryCatch({
      m <- lm(as.formula(formula_str), data = sub)
      coefs <- coef(m)
      coef_list <- c(coef_list, list(coefs))
    }, error = function(e) NULL)
  }

  if (length(coef_list) < 3) return(data.frame())

  # Stack into matrix
  all_vars <- unique(unlist(lapply(coef_list, names)))
  coef_mat <- matrix(NA, nrow = length(coef_list), ncol = length(all_vars))
  colnames(coef_mat) <- all_vars
  for (i in seq_along(coef_list)) {
    coef_mat[i, names(coef_list[[i]])] <- coef_list[[i]]
  }

  # FM average and t-stats (simple, not NW-adjusted for simplicity)
  results <- list()
  for (var in all_vars) {
    vals <- coef_mat[, var]
    vals <- vals[!is.na(vals)]
    if (length(vals) < 3) next
    avg <- mean(vals)
    se  <- sd(vals) / sqrt(length(vals))
    t_stat <- if (se > 0) avg / se else NA_real_
    p_val  <- if (!is.na(t_stat)) {
      2 * pt(abs(t_stat), df = length(vals) - 1, lower.tail = FALSE)
    } else NA_real_

    results <- c(results, list(tibble(
      VARIABLE = var,
      FM_COEFFICIENT = avg,
      FM_STD_ERROR = se,
      FM_T_STAT = t_stat,
      FM_P_VALUE = p_val,
      N_PERIODS = length(vals)
    )))
  }

  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}


# ═══════════════════════════════════════════════════════════════════════
# QUANTILE REGRESSION (tail behavior)
# ═══════════════════════════════════════════════════════════════════════

compute_quantile_regression <- function(panel) {
  #' Quantile regressions at tails and median.
  #'
  #' Tests whether the IS_FUNDAMENTAL effect operates through the tails.
  #'
  #' @param panel data.frame
  #' @return data.frame
  if (!requireNamespace("quantreg", quietly = TRUE)) {
    message("quantreg package required for quantile regression")
    return(data.frame())
  }

  if (nrow(panel) == 0) return(data.frame())
  df <- panel %>%
    filter(HAS_SUFFICIENT_DATA == 1, EVENT_CATEGORY == "FUNDAMENTAL") %>%
    mutate(
      HIGH_CONN_INT = as.integer(HIGH_POLITICAL_CONNECTION),
      HIGH_OPP_INT  = as.integer(HIGH_OPP),
      IS_COURT = as.integer(EVENT_TYPE == "COURT_DECISION")
    )

  valid <- df %>%
    select(ABNORMAL_NET_TRADING, HIGH_CONN_INT, HIGH_OPP_INT, IS_COURT) %>%
    drop_na()
  if (nrow(valid) < 30) return(data.frame())

  results <- list()
  x_cols <- c("HIGH_CONN_INT", "HIGH_OPP_INT", "IS_COURT")

  for (tau in c(0.10, 0.25, 0.50, 0.75, 0.90)) {
    tryCatch({
      m <- quantreg::rq(
        ABNORMAL_NET_TRADING ~ HIGH_CONN_INT + HIGH_OPP_INT + IS_COURT,
        data = valid, tau = tau
      )
      s <- summary(m, se = "boot")
      ct <- as.data.frame(s$coefficients)
      for (var in x_cols) {
        if (var %in% rownames(ct)) {
          results <- c(results, list(tibble(
            QUANTILE = tau, VARIABLE = var,
            COEFFICIENT = ct[var, "Value"],
            STD_ERROR = ct[var, "Std. Error"],
            T_STAT = ct[var, "t value"],
            P_VALUE = ct[var, "Pr(>|t|)"],
            N_OBS = nrow(valid)
          )))
        }
      }
    }, error = function(e) {
      message("QuantReg at tau=", tau, " failed: ", e$message)
    })
  }

  # OLS for comparison
  tryCatch({
    m_ols <- feols(
      ABNORMAL_NET_TRADING ~ HIGH_CONN_INT + HIGH_OPP_INT + IS_COURT,
      data = valid, cluster = ~df$TICKER[match(rownames(valid), rownames(df))]
    )
    ct <- as.data.frame(summary(m_ols)$coeftable)
    ct$VARIABLE <- rownames(ct)
    for (var in x_cols) {
      if (var %in% ct$VARIABLE) {
        row_idx <- which(ct$VARIABLE == var)
        results <- c(results, list(tibble(
          QUANTILE = 0.0,
          VARIABLE = var,
          COEFFICIENT = ct$Estimate[row_idx],
          STD_ERROR = ct$`Std. Error`[row_idx],
          T_STAT = ct$`t value`[row_idx],
          P_VALUE = ct$`Pr(>|t|)`[row_idx],
          N_OBS = nrow(valid)
        )))
      }
    }
  }, error = function(e) {
    message("OLS comparison failed: ", e$message)
  })

  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}


# ═══════════════════════════════════════════════════════════════════════
# JOINT PESARAN-TIMMERMANN 2x2 SIGN TEST
# ═══════════════════════════════════════════════════════════════════════

compute_joint_pt <- function(trades) {
  #' Joint Pesaran-Timmermann (1992) 2x2 sign test.
  #'
  #' Tests buy+sell direction against independence null at the event level.
  #'
  #' @param trades data.frame with TRADE_TYPE, EVENT_CAR, EVENT_ID
  #' @return data.frame (single row) or empty
  if (nrow(trades) == 0) return(data.frame())

  jp_raw <- trades %>%
    filter(!is.na(EVENT_CAR), TRADE_TYPE %in% c("buy", "sell"))
  if (nrow(jp_raw) < 10) return(data.frame())

  # Event-level aggregation: net direction
  jp_ev <- jp_raw %>%
    group_by(EVENT_ID) %>%
    summarise(
      EVENT_CAR = first(EVENT_CAR),
      N_SELLS = sum(TRADE_TYPE == "sell"),
      N_BUYS  = sum(TRADE_TYPE == "buy"),
      .groups = "drop"
    ) %>%
    mutate(NET_DIR = sign(N_SELLS - N_BUYS)) %>%
    filter(NET_DIR != 0) %>%
    mutate(TRADE_DIRECTION = ifelse(NET_DIR > 0, "sell", "buy"))

  n_tot <- nrow(jp_ev)
  if (n_tot < 10) return(data.frame())

  p_sell    <- mean(jp_ev$TRADE_DIRECTION == "sell")
  p_buy     <- 1 - p_sell
  p_neg_car <- mean(jp_ev$EVENT_CAR < 0)
  p_pos_car <- 1 - p_neg_car

  # Joint correct
  correct <- (jp_ev$TRADE_DIRECTION == "sell" & jp_ev$EVENT_CAR < 0) |
    (jp_ev$TRADE_DIRECTION == "buy" & jp_ev$EVENT_CAR > 0)
  obs_acc <- mean(correct)

  # Independence null
  p_star <- p_sell * p_neg_car + p_buy * p_pos_car

  # PT (1992) eq. 7 variance
  var_p_star <- (2 * p_neg_car - 1)^2 * p_sell * (1 - p_sell) / n_tot +
    (2 * p_sell - 1)^2 * p_neg_car * (1 - p_neg_car) / n_tot +
    4 * p_sell * p_neg_car * (1 - p_sell) * (1 - p_neg_car) / n_tot^2

  var_obs <- p_star * (1 - p_star) / n_tot
  var_diff <- var_obs - var_p_star
  z_pt <- if (var_diff > 0) (obs_acc - p_star) / sqrt(var_diff) else NA_real_
  p_pt <- if (!is.na(z_pt)) 2 * pnorm(abs(z_pt), lower.tail = FALSE) else NA_real_

  tibble(
    N_EVENTS = n_tot,
    N_TRADES_RAW = nrow(jp_raw),
    P_SELL = p_sell,
    P_NEG_CAR = p_neg_car,
    NULL_ACCURACY = p_star,
    OBS_ACCURACY = obs_acc,
    EXCESS_ACCURACY = obs_acc - p_star,
    PT_Z_STAT = z_pt,
    PT_P_VALUE = p_pt
  )
}


# ═══════════════════════════════════════════════════════════════════════
# BUILD INSIDER PANEL
# ═══════════════════════════════════════════════════════════════════════

build_insider_panel <- function(form4_data, events, stock_data = NULL) {
  #' Build the insider trading panel for Essay 3.
  #'
  #' Creates event-level observations with abnormal net trading
  #' (pre-event minus benchmark net selling, trading-day normalized).
  #'
  #' @param form4_data data.frame of Form 4 transactions with columns:
  #'   ticker, owner_name, transaction_date, transaction_value, trade_type
  #' @param events data.frame of events with columns:
  #'   EVENT_ID, EVENT_DATE, TICKER, EVENT_CATEGORY, EVENT_TYPE,
  #'   POLICY_AREA (optional), HIGH_POLITICAL_CONNECTION (optional),
  #'   HIGH_OPP (optional)
  #' @param stock_data (unused, placeholder for CRSP data)
  #' @return data.frame panel
  if (nrow(form4_data) == 0 || nrow(events) == 0) return(data.frame())

  # Ensure date types
  form4_data$transaction_date <- as.Date(form4_data$transaction_date)
  events$EVENT_DATE <- as.Date(events$EVENT_DATE)

  # Classify trades if not already done
  if (!("trade_type" %in% names(form4_data)) &&
      "transaction_code" %in% names(form4_data)) {
    form4_data$trade_type <- ifelse(
      form4_data$transaction_code %in% c("S", "D", "F"), "sell",
      ifelse(form4_data$transaction_code %in% c("P", "A"), "buy", "other")
    )
  }

  panel_rows <- list()

  for (i in seq_len(nrow(events))) {
    ev <- events[i, ]
    ticker     <- ev$TICKER
    event_date <- ev$EVENT_DATE

    tkr_txns <- form4_data %>% filter(ticker == !!ticker)
    if (nrow(tkr_txns) == 0) next

    # Pre-event window
    pre_start <- event_date + WINDOWS$PRE_FULL[1]
    pre_end   <- event_date + WINDOWS$PRE_FULL[2]
    pre_txns  <- tkr_txns %>%
      filter(transaction_date >= pre_start, transaction_date <= pre_end)

    # Benchmark window
    bench_start <- event_date + WINDOWS$BENCHMARK[1]
    bench_end   <- event_date + WINDOWS$BENCHMARK[2]
    bench_txns  <- tkr_txns %>%
      filter(transaction_date >= bench_start, transaction_date <= bench_end)

    # Trading-day normalization (252/365)
    pre_cal_days   <- abs(WINDOWS$PRE_FULL[2] - WINDOWS$PRE_FULL[1]) + 1
    bench_cal_days <- abs(WINDOWS$BENCHMARK[2] - WINDOWS$BENCHMARK[1]) + 1
    n_pre_tdays    <- max(1, round(pre_cal_days * 252 / 365))
    n_bench_tdays  <- max(1, round(bench_cal_days * 252 / 365))

    .net_dollar <- function(txns) {
      if (nrow(txns) == 0) return(0)
      sells <- sum(txns$transaction_value[txns$trade_type == "sell"], na.rm = TRUE)
      buys  <- sum(txns$transaction_value[txns$trade_type == "buy"], na.rm = TRUE)
      sells - buys
    }

    pre_net   <- .net_dollar(pre_txns)
    bench_net <- .net_dollar(bench_txns)
    abnormal  <- pre_net / n_pre_tdays - bench_net / n_bench_tdays

    has_sufficient <- nrow(pre_txns) > 0 || nrow(bench_txns) > 0

    row <- tibble(
      EVENT_ID = ev$EVENT_ID,
      TICKER = ticker,
      EVENT_DATE = event_date,
      EVENT_CATEGORY = ev$EVENT_CATEGORY,
      EVENT_TYPE = if ("EVENT_TYPE" %in% names(ev)) ev$EVENT_TYPE else NA_character_,
      POLICY_AREA = if ("POLICY_AREA" %in% names(ev)) ev$POLICY_AREA else NA_character_,
      REGULATORY_PERIOD = assign_regulatory_period(event_date),
      EVENT_YEAR = as.integer(format(event_date, "%Y")),
      HIGH_POLITICAL_CONNECTION = if ("HIGH_POLITICAL_CONNECTION" %in% names(ev)) {
        ev$HIGH_POLITICAL_CONNECTION
      } else FALSE,
      HIGH_OPP = if ("HIGH_OPP" %in% names(ev)) ev$HIGH_OPP else FALSE,
      ABNORMAL_NET_TRADING = abnormal,
      PRE_NET_SELLING = pre_net,
      BENCH_NET_SELLING = bench_net,
      N_PRE_TXNS = nrow(pre_txns),
      N_BENCH_TXNS = nrow(bench_txns),
      HAS_SUFFICIENT_DATA = as.integer(has_sufficient)
    )
    panel_rows <- c(panel_rows, list(row))
  }

  if (length(panel_rows) == 0) return(data.frame())
  bind_rows(panel_rows)
}


# ═══════════════════════════════════════════════════════════════════════
# run_essay3 — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════

run_essay3 <- function(data) {
  #' Run the complete Essay 3 analysis pipeline.
  #'
  #' @param data Named list with components:
  #'   \describe{
  #'     \item{panel}{data.frame from build_insider_panel()}
  #'     \item{form4}{data.frame of Form 4 transactions}
  #'     \item{insider_profits}{(optional) data.frame of insider-level profits}
  #'     \item{trades}{(optional) data.frame of trade-level data with
  #'       EVENT_CAR, EVENT_PROFITABLE, DAYS_BEFORE_EVENT for informed
  #'       trading tests}
  #'   }
  #' @return Named list of all results
  panel <- data$panel
  if (is.null(panel) || nrow(panel) == 0) {
    message("Panel is empty.")
    return(NULL)
  }

  message("=" , strrep("=", 59))
  message("  Essay 3: Insider Trading Around Political Decisions")
  message(strrep("=", 60))

  results <- list()

  # ── 1. Mean vs Distributional ──────────────────────────────────
  message("  1. Mean vs distributional analysis...")
  results$mean_vs_distributional <- compute_mean_vs_distributional(panel)

  # ── 2. Wilcoxon family correction (gating test) ────────────────
  message("  2. Wilcoxon family correction (gating test)...")
  results$wilcoxon_family <- compute_wilcoxon_family(
    results$mean_vs_distributional
  )

  # ── 3. Insider concentration ───────────────────────────────────
  if (!is.null(data$insider_profits) && nrow(data$insider_profits) > 0) {
    message("  3. Insider concentration...")
    results$insider_concentration <- compute_insider_concentration(
      data$insider_profits
    )
  }

  # ── 4. Quantile regression ─────────────────────────────────────
  message("  4. Quantile regression...")
  results$quantile_regression <- compute_quantile_regression(panel)

  # ── 5. Trimmed robustness ──────────────────────────────────────
  message("  5. Trimmed robustness...")
  results$trimmed_robustness <- compute_trimmed_robustness(panel)

  # ── 6. Bootstrap Wilcoxon ──────────────────────────────────────
  message("  6. Bootstrap Wilcoxon...")
  results$bootstrap_wilcoxon <- compute_bootstrap_wilcoxon(panel)

  # ── 7. Insider fixed effects ───────────────────────────────────
  if (!is.null(data$insider_profits) && nrow(data$insider_profits) > 0) {
    message("  7. Insider fixed effects...")
    results$insider_fe <- compute_insider_fe(data$insider_profits)
  }

  # ── 8. TOST equivalence ────────────────────────────────────────
  message("  8. TOST equivalence...")
  results$tost <- compute_tost_equivalence(panel)

  # ── 9. Placebo permutation ─────────────────────────────────────
  message("  9. Placebo permutation...")
  results$placebo <- compute_placebo_test(panel)

  # ── 10. Bootstrap CI ───────────────────────────────────────────
  message("  10. Bootstrap CI...")
  results$bootstrap_ci <- compute_bootstrap_ci(panel)

  # ── 11. Fama-MacBeth ───────────────────────────────────────────
  message("  11. Fama-MacBeth cross-sectional regressions...")
  results$fama_macbeth <- fama_macbeth(panel)

  # ── 12. Informed trading test (if trade-level data provided) ───
  if (!is.null(data$trades) && nrow(data$trades) > 0) {
    message("  12. Informed trading directional accuracy...")
    it_results <- compute_informed_trading(data$trades, panel)
    results$informed_trading   <- it_results$informed_trading
    results$informed_proximity <- it_results$informed_proximity

    # Wild cluster bootstrap on sells
    sells <- data$trades %>%
      filter(TRADE_TYPE == "sell", !is.na(EVENT_PROFITABLE),
             !is.na(EVENT_ID))
    if (nrow(sells) >= 10) {
      message("  12b. Wild cluster bootstrap (event-clustered)...")
      results$cluster_inference <- wild_cluster_bootstrap_proportion(
        sells$EVENT_PROFITABLE, sells$EVENT_ID,
        p0 = 0.5, n_boot = 1000, seed = 42
      )
    }

    # Joint Pesaran-Timmermann
    message("  12c. Joint Pesaran-Timmermann sign test...")
    results$joint_pt <- compute_joint_pt(data$trades)

    # Date-randomization placebo
    message("  12d. Directional accuracy placebo...")
    results$directional_placebo <- placebo_directional(data$trades)
  }

  # ── Summary ────────────────────────────────────────────────────
  message(strrep("=", 60))
  n_cult <- sum(panel$EVENT_CATEGORY == "CULTURAL", na.rm = TRUE)
  n_fund <- sum(panel$EVENT_CATEGORY == "FUNDAMENTAL", na.rm = TRUE)
  message(sprintf("  Panel: %d events (%d cultural, %d fundamental)",
                   nrow(panel), n_cult, n_fund))

  wf <- results$wilcoxon_family
  if (!is.null(wf) && nrow(wf) > 0) {
    n_bh   <- sum(wf$BH_SIGNIFICANT, na.rm = TRUE)
    n_holm <- sum(wf$HOLM_SIGNIFICANT, na.rm = TRUE)
    message(sprintf("  Gating test: %d BH-significant, %d Holm-significant",
                     n_bh, n_holm))
  }

  tost <- results$tost
  if (!is.null(tost) && nrow(tost) > 0) {
    for (i in seq_len(nrow(tost))) {
      message(sprintf("  TOST [%s]: P_TOST=%.4f, equivalent=%s",
                       tost$EVENT_CATEGORY[i], tost$P_TOST[i],
                       tost$EQUIVALENT[i]))
    }
  }

  message(strrep("=", 60))

  results$panel <- panel
  results
}
