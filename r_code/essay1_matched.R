# =============================================================================
# Essay 1 -- Matched Control Analysis
#
# Full Essay 1 pipeline extension: treatment vs control FF5 comparison
# across volatility regimes using industry-matched control firms.
#
# References:
#   Barillas, F. & Shanken, J. (2017). Review of Financial Studies, 30(4).
#   Fama, E.F. & French, K.R. (2015). J. Financial Economics, 116(1).
#
# Author: Ashley Roseboro
# =============================================================================

library(tidyverse)
library(sandwich)
library(lmtest)
library(broom)

source(file.path(tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "."), "essay1.R"))

# -- Constants ----------------------------------------------------------------
BH_FDR_Q <- 0.10

# =============================================================================
# Internal: Run FF5 regression for a single stock within each regime
# =============================================================================
#' @param ticker character, stock ticker
#' @param stock_returns data.frame with DATE, TICKER, RETURN
#' @param factor_regime data.frame with DATE, FF5 factors, RF, REGIME_LABEL
#' @param regime_labels character vector of regime labels (sorted)
#' @return named list of lists, keyed by regime label
.run_stock_ff5 <- function(ticker, stock_returns, factor_regime, regime_labels) {
  ret <- stock_returns %>%
    filter(TICKER == ticker) %>%
    mutate(DATE = as.Date(DATE)) %>%
    select(DATE, RETURN) %>%
    drop_na()

  if (nrow(ret) == 0) return(list())

  # Convert percent to decimal if needed
  if (max(abs(ret$RETURN), na.rm = TRUE) > 1.5) {
    ret$RETURN <- ret$RETURN / 100
  }

  merged <- ret %>%
    inner_join(factor_regime, by = "DATE") %>%
    mutate(EXCESS_RETURN = RETURN - RF) %>%
    drop_na(c("EXCESS_RETURN", FF5_ALL))

  results <- list()
  for (label in regime_labels) {
    sub <- merged %>% filter(REGIME_LABEL == label)
    n_obs <- nrow(sub)

    if (n_obs < MIN_REGIME_OBS) {
      results[[label]] <- list(
        ticker = ticker, regime = label, sufficient_obs = FALSE,
        n_obs = n_obs, alpha = NA_real_, alpha_t = NA_real_,
        alpha_p = NA_real_, r_squared = NA_real_,
        betas = list(), t_stats = list(), p_values = list()
      )
      next
    }

    tryCatch({
      fml <- as.formula(paste("EXCESS_RETURN ~", paste(FF5_ALL, collapse = " + ")))
      fit <- lm(fml, data = sub)
      nw <- coeftest(fit, vcov = NeweyWest(fit, lag = HAC_MAXLAGS, prewhite = FALSE))

      betas    <- setNames(as.list(nw[FF5_ALL, 1]), FF5_ALL)
      t_stats  <- setNames(as.list(nw[FF5_ALL, 3]), FF5_ALL)
      p_values <- setNames(as.list(nw[FF5_ALL, 4]), FF5_ALL)

      results[[label]] <- list(
        ticker        = ticker,
        regime        = label,
        sufficient_obs = TRUE,
        n_obs         = as.integer(nobs(fit)),
        alpha         = nw["(Intercept)", 1],
        alpha_t       = nw["(Intercept)", 3],
        alpha_p       = nw["(Intercept)", 4],
        r_squared     = summary(fit)$r.squared,
        betas         = betas,
        t_stats       = t_stats,
        p_values      = p_values
      )
    }, error = function(e) {
      results[[label]] <<- list(
        ticker = ticker, regime = label, sufficient_obs = FALSE,
        n_obs = n_obs, alpha = NA_real_, alpha_t = NA_real_,
        alpha_p = NA_real_, r_squared = NA_real_,
        betas = list(), t_stats = list(), p_values = list()
      )
    })
  }
  return(results)
}

# =============================================================================
# Build Result Rows from Ticker Regression Results
# =============================================================================
.build_result_rows <- function(tickers, group_label, ticker_results,
                                pricing_factors, labels) {
  rows <- list()
  for (ticker in tickers) {
    res <- ticker_results[[ticker]]
    if (is.null(res)) next
    for (label in labels) {
      sr <- res[[label]]
      if (is.null(sr) || !sr$sufficient_obs) next
      row <- data.frame(
        TICKER    = ticker,
        GROUP     = group_label,
        REGIME    = label,
        N_OBS     = sr$n_obs,
        ALPHA     = sr$alpha,
        ALPHA_T   = sr$alpha_t,
        ALPHA_P   = sr$alpha_p,
        R_SQUARED = sr$r_squared,
        stringsAsFactors = FALSE
      )
      for (f in pricing_factors) {
        row[[paste0(f, "_BETA")]] <- sr$betas[[f]] %||% NA_real_
        row[[paste0(f, "_T")]]    <- sr$t_stats[[f]] %||% NA_real_
        row[[paste0(f, "_P")]]    <- sr$p_values[[f]] %||% NA_real_
      }
      rows[[length(rows) + 1]] <- row
    }
  }
  if (length(rows) == 0) return(data.frame())
  bind_rows(rows)
}

# =============================================================================
# Matched Control FF5 Analysis
# =============================================================================
#' Compare FF5 factor loadings between culture war firms and
#' industry-matched control firms across volatility regimes.
#'
#' Tests:
#'   1. Paired t-test: mean(delta_beta) = 0 per factor per regime
#'   2. Regime amplification: delta(HighVol) - delta(LowVol) per factor
#'   3. Sign consistency: percent of pairs with same-sign delta
#'
#' @param stock_data data.frame with DATE, TICKER, RETURN
#' @param cw_data data.frame with culture war event tickers
#' @param control_data data.frame with TREATMENT_TICKER, CONTROL_TICKER columns
#' @param regime_result output from estimate_vix_regimes()
#' @param ff_factors data.frame with DATE, FF5 factors, RF
#' @return named list (MatchedControlResult equivalent):
#'   - treatment_results: data.frame
#'   - control_results: data.frame
#'   - delta_betas: data.frame
#'   - paired_ttest: data.frame
#'   - regime_amplification: data.frame
#'   - sign_consistency: data.frame
#'   - n_pairs: integer
#'   - n_pairs_complete: integer
#'   - coverage: data.frame
ff5_matched_control_analysis <- function(stock_data, cw_data, control_data,
                                          regime_result, ff_factors = NULL) {
  if (is.null(regime_result)) {
    warning("No regime result provided")
    return(NULL)
  }
  if (nrow(control_data) == 0) {
    warning("No control companies data")
    return(NULL)
  }

  pricing_factors <- FF5_ALL

  # Prepare factors
  if (is.null(ff_factors)) {
    warning("ff_factors must be provided")
    return(NULL)
  }

  factors <- ff_factors %>%
    mutate(DATE = as.Date(DATE)) %>%
    select(DATE, all_of(pricing_factors), RF) %>%
    drop_na()

  for (col in c(pricing_factors, "RF")) {
    if (max(abs(factors[[col]]), na.rm = TRUE) > 1.5) {
      factors[[col]] <- factors[[col]] / 100
    }
  }

  # Merge factors with regime assignments
  factor_regime <- factors %>%
    inner_join(
      regime_result$regime_assignments %>% select(DATE, REGIME_LABEL),
      by = "DATE"
    )

  labels <- names(sort(regime_result$regime_means))

  # Collect unique tickers
  all_treatments <- unique(control_data$TREATMENT_TICKER)
  all_controls   <- unique(control_data$CONTROL_TICKER)
  all_tickers    <- unique(c(all_treatments, all_controls))

  message("Running matched control FF5: ", length(all_treatments), " treatment, ",
          length(all_controls), " control, ", length(all_tickers), " unique tickers")

  # Run regressions for all unique tickers
  ticker_results <- list()
  for (ticker in all_tickers) {
    ticker_results[[ticker]] <- .run_stock_ff5(ticker, stock_data, factor_regime, labels)
  }

  treatment_df <- .build_result_rows(all_treatments, "TREATMENT", ticker_results,
                                      pricing_factors, labels)
  control_df   <- .build_result_rows(all_controls, "CONTROL", ticker_results,
                                      pricing_factors, labels)

  # Compute deltas at the pair level
  delta_rows <- list()
  for (i in seq_len(nrow(control_data))) {
    treat_ticker <- control_data$TREATMENT_TICKER[i]
    ctrl_ticker  <- control_data$CONTROL_TICKER[i]
    treat_res    <- ticker_results[[treat_ticker]]
    ctrl_res     <- ticker_results[[ctrl_ticker]]

    for (label in labels) {
      t_r <- treat_res[[label]]
      c_r <- ctrl_res[[label]]
      if (is.null(t_r) || is.null(c_r) || !t_r$sufficient_obs || !c_r$sufficient_obs) next

      row <- data.frame(
        TREATMENT_TICKER = treat_ticker,
        CONTROL_TICKER   = ctrl_ticker,
        REGIME           = label,
        ALPHA_DELTA      = t_r$alpha - c_r$alpha,
        R_SQUARED_TREAT  = t_r$r_squared,
        R_SQUARED_CTRL   = c_r$r_squared,
        stringsAsFactors  = FALSE
      )
      for (f in pricing_factors) {
        row[[paste0(f, "_DELTA")]] <- (t_r$betas[[f]] %||% NA_real_) -
                                       (c_r$betas[[f]] %||% NA_real_)
        row[[paste0(f, "_TREAT")]] <- t_r$betas[[f]] %||% NA_real_
        row[[paste0(f, "_CTRL")]]  <- c_r$betas[[f]] %||% NA_real_
      }
      delta_rows[[length(delta_rows) + 1]] <- row
    }
  }

  delta_df <- if (length(delta_rows) > 0) bind_rows(delta_rows) else data.frame()
  n_pairs <- nrow(control_data)

  if (nrow(delta_df) == 0) {
    warning("No matched pairs produced delta results")
    return(NULL)
  }

  # Count complete pairs (results in all regimes)
  pair_counts <- delta_df %>%
    group_by(TREATMENT_TICKER, CONTROL_TICKER) %>%
    summarise(N_REGIMES = n_distinct(REGIME), .groups = "drop")
  n_complete <- sum(pair_counts$N_REGIMES == length(labels))

  # --- Test 1: Paired t-test per factor per regime ---
  delta_cols <- c(paste0(pricing_factors, "_DELTA"), "ALPHA_DELTA")
  ttest_rows <- list()
  for (label in labels) {
    sub <- delta_df %>% filter(REGIME == label)
    if (nrow(sub) < 5) next
    for (dc in delta_cols) {
      # Aggregate to one delta per treatment firm
      agg <- sub %>%
        group_by(TREATMENT_TICKER) %>%
        summarise(DELTA = mean(.data[[dc]], na.rm = TRUE), .groups = "drop") %>%
        filter(!is.na(DELTA))
      if (nrow(agg) < 5) next

      tt <- t.test(agg$DELTA, mu = 0)
      ttest_rows[[length(ttest_rows) + 1]] <- data.frame(
        REGIME            = label,
        VARIABLE          = gsub("_DELTA$", "", dc),
        MEAN_DELTA        = mean(agg$DELTA),
        STD_DELTA         = sd(agg$DELTA),
        N_TREATMENT_FIRMS = nrow(agg),
        N_RAW_PAIRS       = sum(!is.na(sub[[dc]])),
        T_STAT            = tt$statistic,
        P_VALUE           = tt$p.value,
        stringsAsFactors   = FALSE
      )
    }
  }
  paired_ttest <- if (length(ttest_rows) > 0) bind_rows(ttest_rows) else data.frame()

  # --- Test 2: Regime amplification (High - Low) ---
  did_rows <- list()
  if (length(labels) >= 2) {
    low_label  <- labels[1]
    high_label <- labels[length(labels)]
    low_deltas  <- delta_df %>% filter(REGIME == low_label)
    high_deltas <- delta_df %>% filter(REGIME == high_label)

    for (dc in delta_cols) {
      low_vals  <- low_deltas %>%
        select(TREATMENT_TICKER, CONTROL_TICKER, LOW = all_of(dc))
      high_vals <- high_deltas %>%
        select(TREATMENT_TICKER, CONTROL_TICKER, HIGH = all_of(dc))
      merged_did <- inner_join(low_vals, high_vals,
                                by = c("TREATMENT_TICKER", "CONTROL_TICKER"))
      if (nrow(merged_did) < 5) next

      agg <- merged_did %>%
        group_by(TREATMENT_TICKER) %>%
        summarise(LOW = mean(LOW, na.rm = TRUE),
                  HIGH = mean(HIGH, na.rm = TRUE),
                  .groups = "drop") %>%
        mutate(DIFF = HIGH - LOW) %>%
        filter(!is.na(DIFF))
      if (nrow(agg) < 5) next

      tt <- t.test(agg$DIFF, mu = 0)
      did_rows[[length(did_rows) + 1]] <- data.frame(
        VARIABLE          = gsub("_DELTA$", "", dc),
        MEAN_DELTA_LOW    = mean(agg$LOW),
        MEAN_DELTA_HIGH   = mean(agg$HIGH),
        MEAN_DIFF         = mean(agg$DIFF),
        N_TREATMENT_FIRMS = nrow(agg),
        N_RAW_PAIRS       = nrow(merged_did),
        T_STAT            = tt$statistic,
        P_VALUE           = tt$p.value,
        stringsAsFactors   = FALSE
      )
    }
  }
  regime_amplification <- if (length(did_rows) > 0) bind_rows(did_rows) else data.frame()

  # --- Test 3: Sign consistency ---
  sign_rows <- list()
  for (label in labels) {
    sub <- delta_df %>% filter(REGIME == label)
    if (nrow(sub) == 0) next
    for (dc in delta_cols) {
      agg <- sub %>%
        group_by(TREATMENT_TICKER) %>%
        summarise(DELTA = mean(.data[[dc]], na.rm = TRUE), .groups = "drop") %>%
        filter(!is.na(DELTA))
      if (nrow(agg) < 5) next

      n_pos   <- sum(agg$DELTA > 0)
      n_neg   <- sum(agg$DELTA < 0)
      n_total <- nrow(agg)
      majority_sign <- if (n_pos >= n_neg) "positive" else "negative"
      consistency   <- max(n_pos, n_neg) / n_total
      binom_p       <- binom.test(n_pos, n_total, p = 0.5)$p.value

      sign_rows[[length(sign_rows) + 1]] <- data.frame(
        REGIME            = label,
        VARIABLE          = gsub("_DELTA$", "", dc),
        N_POSITIVE        = n_pos,
        N_NEGATIVE        = n_neg,
        N_TREATMENT_FIRMS = n_total,
        PCT_MAJORITY      = consistency,
        MAJORITY_SIGN     = majority_sign,
        BINOMIAL_P        = binom_p,
        stringsAsFactors   = FALSE
      )
    }
  }
  sign_consistency <- if (length(sign_rows) > 0) bind_rows(sign_rows) else data.frame()

  # BH correction
  if (nrow(paired_ttest) > 0 && "P_VALUE" %in% names(paired_ttest)) {
    paired_ttest$BH_SIGNIFICANT <- benjamini_hochberg(paired_ttest$P_VALUE, q = BH_FDR_Q)
  }
  if (nrow(regime_amplification) > 0 && "P_VALUE" %in% names(regime_amplification)) {
    regime_amplification$BH_SIGNIFICANT <- benjamini_hochberg(
      regime_amplification$P_VALUE, q = BH_FDR_Q)
  }
  if (nrow(sign_consistency) > 0 && "BINOMIAL_P" %in% names(sign_consistency)) {
    sign_consistency$BH_SIGNIFICANT <- benjamini_hochberg(
      sign_consistency$BINOMIAL_P, q = BH_FDR_Q)
  }

  # Coverage table
  coverage_rows <- list()
  for (ticker in names(ticker_results)) {
    res <- ticker_results[[ticker]]
    for (label in labels) {
      sr <- res[[label]]
      status <- if (is.null(sr)) "NO_DATA"
                else if (sr$sufficient_obs) "OK"
                else if (sr$n_obs < MIN_REGIME_OBS) "INSUFFICIENT_OBS"
                else "REGRESSION_FAILED"
      coverage_rows[[length(coverage_rows) + 1]] <- data.frame(
        TICKER     = ticker,
        REGIME     = label,
        HAS_RESULT = !is.null(sr) && sr$sufficient_obs,
        STATUS     = status,
        N_OBS      = if (!is.null(sr)) sr$n_obs else 0L,
        stringsAsFactors = FALSE
      )
    }
  }
  coverage_df <- if (length(coverage_rows) > 0) bind_rows(coverage_rows) else data.frame()

  list(
    treatment_results    = treatment_df,
    control_results      = control_df,
    delta_betas          = delta_df,
    paired_ttest         = paired_ttest,
    regime_amplification = regime_amplification,
    sign_consistency     = sign_consistency,
    n_pairs              = n_pairs,
    n_pairs_complete     = as.integer(n_complete),
    coverage             = coverage_df
  )
}
