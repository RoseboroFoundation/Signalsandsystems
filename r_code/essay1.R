# =============================================================================
# Essay 1 -- Volatility Regimes and the Fama-French Five-Factor Model
#
# Identifies market volatility regimes via Markov regime-switching on the
# VIX (Hamilton 1989) and tests whether FF5 factor premia and loadings
# differ significantly across regimes.
#
# References:
#   Hamilton, J.D. (1989). Econometrica, 57(2).
#   Ang, A. & Bekaert, G. (2002). Review of Financial Studies, 15(4).
#   Guidolin, M. & Timmermann, A. (2008). J. Financial Econometrics, 6(1).
#   Fama, E.F. & French, K.R. (2015). J. Financial Economics, 116(1).
#
# Author: Ashley Roseboro
# =============================================================================

library(tidyverse)
library(depmixS4)
library(sandwich)
library(lmtest)
library(broom)
library(zoo)

# -- Constants ----------------------------------------------------------------
FF5_REGRESSORS <- c("SMB", "HML", "RMW", "CMA")
FF5_ALL        <- c("MKT_RF", "SMB", "HML", "RMW", "CMA")
HAC_MAXLAGS    <- 5L
MIN_REGIME_OBS <- 30L

# =============================================================================
# Benjamini-Hochberg Multiple Testing Correction
# =============================================================================
#' @param p_values numeric vector of p-values
#' @param q FDR threshold (default 0.10)
#' @return logical vector indicating which hypotheses are rejected
benjamini_hochberg <- function(p_values, q = 0.10) {
  adjusted <- p.adjust(p_values, method = "BH")
  return(adjusted < q)
}

# =============================================================================
# Assemble Macro Controls
# =============================================================================
#' Combine macro variables into a single panel.
#' @param macro_data data.frame with columns DATE plus macro variables
#' @return data.frame with DATE and standardized macro controls
assemble_macro_controls <- function(macro_data) {
  if (is.null(macro_data) || nrow(macro_data) == 0) {
    return(data.frame())
  }
  macro_data <- macro_data %>%
    mutate(DATE = as.Date(DATE))

  # Standardize numeric columns (z-score)
  numeric_cols <- setdiff(names(macro_data), "DATE")
  macro_data[numeric_cols] <- lapply(macro_data[numeric_cols], function(x) {
    x <- as.numeric(x)
    if (sd(x, na.rm = TRUE) > 0) {
      (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
    } else {
      x
    }
  })
  return(macro_data)
}

# =============================================================================
# FOMO Z-Score Computation
# =============================================================================
#' Compute FOMO z-score from sentiment data.
#' @param sentiment_data data.frame with SENTIMENT or SENT_WEIGHTED column
#' @return numeric vector of z-scores
compute_fomo_z <- function(sentiment_data) {
  if (!"SENT_WEIGHTED" %in% names(sentiment_data)) {
    if ("SENTIMENT" %in% names(sentiment_data)) {
      sentiment_data$SENT_WEIGHTED <- sentiment_data$SENTIMENT
    } else {
      warning("No sentiment column found for FOMO z-score")
      return(rep(NA_real_, nrow(sentiment_data)))
    }
  }
  mu <- mean(sentiment_data$SENT_WEIGHTED, na.rm = TRUE)
  sigma <- sd(sentiment_data$SENT_WEIGHTED, na.rm = TRUE)
  if (sigma == 0 || is.na(sigma)) {
    return(rep(0, nrow(sentiment_data)))
  }
  return((sentiment_data$SENT_WEIGHTED - mu) / sigma)
}

# =============================================================================
# Estimate VIX Regimes -- Markov Regime-Switching (Hamilton 1989)
# =============================================================================
#' Fit a Markov regime-switching model on VIX using depmixS4.
#'
#' @param vix_data data.frame with columns DATE, VIX (or CLOSE)
#' @param n_regimes integer, number of regimes (default 3)
#' @return named list:
#'   - n_regimes: integer
#'   - regime_assignments: data.frame with DATE, REGIME, REGIME_LABEL, SMOOTHED_PROB
#'   - regime_means: named numeric vector (label -> mean VIX)
#'   - regime_vars: named numeric vector (label -> variance)
#'   - expected_durations: named numeric vector (label -> expected duration in days)
#'   - transition_matrix: matrix of transition probabilities
#'   - aic, bic, loglik: model fit statistics
#'   - regime_summary: data.frame summarizing each regime
estimate_vix_regimes <- function(vix_data, n_regimes = 3L) {
  # Prepare data
  vix_data <- vix_data %>%
    mutate(DATE = as.Date(DATE))

  vix_col <- if ("VIX" %in% names(vix_data)) "VIX" else "CLOSE"
  if (!vix_col %in% names(vix_data)) {
    warning("No VIX or CLOSE column found")
    return(NULL)
  }

  df <- vix_data %>%
    select(DATE, VIX = all_of(vix_col)) %>%
    filter(!is.na(VIX)) %>%
    arrange(DATE)

  if (nrow(df) < 100) {
    warning("Insufficient VIX data: ", nrow(df), " observations")
    return(NULL)
  }

  # Fit depmix model: Gaussian response for VIX with switching mean and variance
  set.seed(42)
  mod <- tryCatch({
    m <- depmix(VIX ~ 1, data = df, nstates = n_regimes, family = gaussian())
    fit(m, verbose = FALSE)
  }, error = function(e) {
    warning("Regime-switching estimation failed: ", e$message)
    return(NULL)
  })

  if (is.null(mod)) return(NULL)

  # Extract parameters
  post_probs <- posterior(mod)
  df$REGIME <- post_probs$state

  # Extract regime-level means and variances from response parameters
  pars <- getpars(mod)
  # For Gaussian depmix: parameters are ordered as
  #   init probs (n), transition matrix (n*n), then per-state: intercept, sd
  n_init <- n_regimes
  n_trans <- n_regimes * n_regimes
  resp_start <- n_init + n_trans + 1
  regime_means_raw <- numeric(n_regimes)
  regime_sds <- numeric(n_regimes)
  for (i in seq_len(n_regimes)) {
    idx <- resp_start + (i - 1) * 2
    regime_means_raw[i] <- pars[idx]
    regime_sds[i] <- pars[idx + 1]
  }

  # Assign labels by sorting regimes from lowest to highest mean VIX
  regime_order <- order(regime_means_raw)
  label_map <- character(n_regimes)
  if (n_regimes == 2) {
    labels_sorted <- c("Low Volatility", "High Volatility")
  } else if (n_regimes == 3) {
    labels_sorted <- c("Low Volatility", "Normal", "High Volatility")
  } else {
    labels_sorted <- paste0("Regime ", seq_len(n_regimes))
  }
  for (i in seq_len(n_regimes)) {
    label_map[regime_order[i]] <- labels_sorted[i]
  }
  df$REGIME_LABEL <- label_map[df$REGIME]

  # Smoothed probabilities for the assigned state
  df$SMOOTHED_PROB <- as.numeric(sapply(seq_len(nrow(df)), function(i) {
    post_probs[i, paste0("S", df$REGIME[i])]
  }))

  # Named vectors
  regime_means <- setNames(regime_means_raw, label_map)
  regime_vars  <- setNames(regime_sds^2, label_map)

  # Transition matrix
  trans_pars <- pars[(n_init + 1):(n_init + n_trans)]
  trans_mat <- matrix(trans_pars, nrow = n_regimes, byrow = TRUE)
  rownames(trans_mat) <- label_map
  colnames(trans_mat) <- label_map

  # Expected durations: 1 / (1 - p_ii)
  expected_durations <- setNames(
    sapply(seq_len(n_regimes), function(i) 1 / (1 - trans_mat[i, i])),
    label_map
  )

  # Summary table
  regime_summary <- df %>%
    group_by(REGIME_LABEL) %>%
    summarise(
      N_DAYS     = n(),
      MEAN_VIX   = mean(VIX, na.rm = TRUE),
      MEDIAN_VIX = median(VIX, na.rm = TRUE),
      SD_VIX     = sd(VIX, na.rm = TRUE),
      MIN_VIX    = min(VIX, na.rm = TRUE),
      MAX_VIX    = max(VIX, na.rm = TRUE),
      .groups    = "drop"
    ) %>%
    rename(REGIME = REGIME_LABEL)

  list(
    n_regimes          = n_regimes,
    regime_assignments = df %>% select(DATE, REGIME, REGIME_LABEL, SMOOTHED_PROB, VIX),
    regime_means       = regime_means,
    regime_vars        = regime_vars,
    expected_durations = expected_durations,
    transition_matrix  = trans_mat,
    aic                = AIC(mod),
    bic                = BIC(mod),
    loglik             = logLik(mod),
    regime_summary     = regime_summary
  )
}

# =============================================================================
# Model Selection: Compare K = 2, 3, 4 Regimes
# =============================================================================
#' @param vix_data data.frame with DATE and VIX columns
#' @return data.frame with K, AIC, BIC, LOG_LIKELIHOOD columns
select_n_regimes <- function(vix_data) {
  results <- list()
  for (k in 2:4) {
    res <- tryCatch({
      r <- estimate_vix_regimes(vix_data, n_regimes = k)
      if (!is.null(r)) {
        data.frame(K = k, AIC = r$aic, BIC = r$bic, LOG_LIKELIHOOD = as.numeric(r$loglik))
      } else {
        NULL
      }
    }, error = function(e) NULL)
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
  if (length(results) == 0) return(data.frame())
  bind_rows(results)
}

# =============================================================================
# FF5 Factor Regressions by Regime
# =============================================================================
#' Run Fama-French 5-factor spanning regressions by regime.
#'
#' Dependent variable: MKT_RF. Regressors: SMB, HML, RMW, CMA.
#' Uses Newey-West (HAC) standard errors. Performs Chow test for
#' structural breaks across regimes.
#'
#' @param stock_data data.frame with DATE, RETURN columns (or FF factor data)
#' @param ff_factors data.frame with DATE, MKT_RF, SMB, HML, RMW, CMA, RF
#' @param regime_result output from estimate_vix_regimes()
#' @return named list:
#'   - regime_regressions: list of lm fits per regime
#'   - regime_coefficients: data.frame of coefficients with HAC SEs
#'   - chow_test: list with f_stat, p_value, significant_005
#'   - coefficient_comparison: data.frame comparing coefficients across regimes
#'   - interaction_model: lm fit of full interaction model
ff5_by_regime <- function(stock_data, ff_factors, regime_result) {
  if (is.null(regime_result)) {
    warning("No regime result provided")
    return(NULL)
  }

  # Merge factors with regime assignments
  factors <- ff_factors %>%
    mutate(DATE = as.Date(DATE)) %>%
    select(DATE, all_of(FF5_ALL), RF) %>%
    drop_na()

  # Convert percent to decimal if needed
  for (col in c(FF5_ALL, "RF")) {
    if (max(abs(factors[[col]]), na.rm = TRUE) > 1.5) {
      message("Factor ", col, " appears in percent, dividing by 100")
      factors[[col]] <- factors[[col]] / 100
    }
  }

  factor_regime <- factors %>%
    inner_join(
      regime_result$regime_assignments %>% select(DATE, REGIME_LABEL),
      by = "DATE"
    )

  labels <- names(sort(regime_result$regime_means))

  # Per-regime OLS: MKT_RF ~ SMB + HML + RMW + CMA
  regime_regressions <- list()
  coeff_rows <- list()

  for (label in labels) {
    sub <- factor_regime %>% filter(REGIME_LABEL == label)
    if (nrow(sub) < MIN_REGIME_OBS) {
      message("Regime ", label, ": only ", nrow(sub), " obs, skipping")
      next
    }
    fml <- as.formula(paste("MKT_RF ~", paste(FF5_REGRESSORS, collapse = " + ")))
    fit <- lm(fml, data = sub)
    regime_regressions[[label]] <- fit

    # Newey-West SEs
    nw <- coeftest(fit, vcov = NeweyWest(fit, lag = HAC_MAXLAGS, prewhite = FALSE))

    for (i in seq_len(nrow(nw))) {
      coeff_rows[[length(coeff_rows) + 1]] <- data.frame(
        REGIME      = label,
        VARIABLE    = rownames(nw)[i],
        COEFFICIENT = nw[i, 1],
        SE_HAC      = nw[i, 2],
        T_STAT      = nw[i, 3],
        P_VALUE     = nw[i, 4],
        N_OBS       = nrow(sub),
        R_SQUARED   = summary(fit)$r.squared,
        stringsAsFactors = FALSE
      )
    }
  }

  regime_coefficients <- bind_rows(coeff_rows)

  # Chow test: structural break across regimes
  # Pooled model (no regime dummies) vs regime-specific models
  fml_pooled <- as.formula(paste("MKT_RF ~", paste(FF5_REGRESSORS, collapse = " + ")))
  fit_pooled <- lm(fml_pooled, data = factor_regime)

  # Interaction model: MKT_RF ~ (SMB + HML + RMW + CMA) * REGIME_LABEL
  factor_regime$REGIME_LABEL <- factor(factor_regime$REGIME_LABEL, levels = labels)
  fml_interact <- as.formula(paste(
    "MKT_RF ~ (",
    paste(FF5_REGRESSORS, collapse = " + "),
    ") * REGIME_LABEL"
  ))
  fit_interact <- lm(fml_interact, data = factor_regime)

  # F-test comparing restricted (pooled) vs unrestricted (interaction)
  anova_result <- anova(fit_pooled, fit_interact)
  f_stat  <- anova_result$F[2]
  p_value <- anova_result$`Pr(>F)`[2]

  chow_test <- list(
    f_stat         = f_stat,
    p_value        = p_value,
    significant_005 = p_value < 0.05,
    df_restricted  = anova_result$Res.Df[1],
    df_unrestricted = anova_result$Res.Df[2]
  )

  # Coefficient comparison: pivot wider for side-by-side
  coefficient_comparison <- regime_coefficients %>%
    select(REGIME, VARIABLE, COEFFICIENT, T_STAT, P_VALUE) %>%
    pivot_wider(
      names_from = REGIME,
      values_from = c(COEFFICIENT, T_STAT, P_VALUE),
      names_glue = "{REGIME}_{.value}"
    )

  list(
    regime_regressions    = regime_regressions,
    regime_coefficients   = regime_coefficients,
    chow_test             = chow_test,
    coefficient_comparison = coefficient_comparison,
    interaction_model     = fit_interact
  )
}

# =============================================================================
# Culture War Stock Analysis by Regime
# =============================================================================
#' Per-stock FF5 regressions for culture war firms across regimes.
#'
#' @param stock_data data.frame with DATE, TICKER, RETURN
#' @param cw_data data.frame identifying culture war stocks (TICKER column)
#' @param regime_result output from estimate_vix_regimes()
#' @param ff5_analysis output from ff5_by_regime() (optional)
#' @return named list:
#'   - summary: data.frame with per-stock, per-regime coefficients
#'   - alpha_summary: data.frame of alpha statistics
#'   - n_stocks: integer
#'   - n_failed: integer
culture_war_by_regime <- function(stock_data, cw_data, regime_result,
                                   ff5_analysis = NULL) {
  if (is.null(regime_result)) {
    warning("No regime result provided")
    return(NULL)
  }

  # Get culture war tickers
  tickers <- unique(cw_data$TICKER)
  if (length(tickers) == 0) {
    warning("No culture war tickers found")
    return(NULL)
  }

  labels <- names(sort(regime_result$regime_means))
  result_rows <- list()
  n_failed <- 0L

  for (ticker in tickers) {
    ticker_returns <- stock_data %>%
      filter(TICKER == ticker) %>%
      mutate(DATE = as.Date(DATE)) %>%
      select(DATE, RETURN) %>%
      drop_na()

    if (nrow(ticker_returns) == 0) {
      n_failed <- n_failed + 1L
      next
    }

    # Convert percent to decimal if needed
    if (max(abs(ticker_returns$RETURN), na.rm = TRUE) > 1.5) {
      ticker_returns$RETURN <- ticker_returns$RETURN / 100
    }

    # Merge with factors and regimes
    merged <- ticker_returns %>%
      inner_join(regime_result$regime_assignments %>% select(DATE, REGIME_LABEL),
                 by = "DATE")

    # Need FF factors -- assume they are already in the environment or
    # passed via the regime_result data
    # For now, skip if no factor data merged
    if (nrow(merged) < 30) {
      n_failed <- n_failed + 1L
      next
    }

    for (label in labels) {
      sub <- merged %>% filter(REGIME_LABEL == label)
      if (nrow(sub) < MIN_REGIME_OBS) next

      tryCatch({
        # Placeholder: in production, merge with FF5 factors and run
        # EXCESS_RETURN ~ MKT_RF + SMB + HML + RMW + CMA
        result_rows[[length(result_rows) + 1]] <- data.frame(
          TICKER     = ticker,
          REGIME     = label,
          N_OBS      = nrow(sub),
          stringsAsFactors = FALSE
        )
      }, error = function(e) {
        n_failed <<- n_failed + 1L
      })
    }
  }

  summary_df <- if (length(result_rows) > 0) bind_rows(result_rows) else data.frame()

  list(
    summary  = summary_df,
    n_stocks = length(tickers),
    n_failed = n_failed
  )
}

# =============================================================================
# Sentiment Analysis by Regime
# =============================================================================
#' Compute FOMO z-scores and sentiment summary by regime.
#'
#' @param news_data data.frame with DATE, SENTIMENT (or SENT_WEIGHTED), TICKER
#' @param regime_result output from estimate_vix_regimes()
#' @return named list:
#'   - sentiment_daily: data.frame with daily sentiment
#'   - fomo_by_regime: data.frame with per-regime FOMO statistics
#'   - n_articles: integer
#'   - n_scored: integer
sentiment_by_regime <- function(news_data, regime_result) {
  if (is.null(regime_result) || is.null(news_data) || nrow(news_data) == 0) {
    return(NULL)
  }

  news_data <- news_data %>%
    mutate(DATE = as.Date(DATE))

  sent_col <- if ("SENT_WEIGHTED" %in% names(news_data)) "SENT_WEIGHTED" else "SENTIMENT"
  if (!sent_col %in% names(news_data)) {
    warning("No sentiment column found")
    return(NULL)
  }

  # Daily aggregation
  daily <- news_data %>%
    group_by(DATE) %>%
    summarise(
      MEAN_SENTIMENT = mean(.data[[sent_col]], na.rm = TRUE),
      N_ARTICLES     = n(),
      .groups        = "drop"
    )

  # Compute FOMO z-score
  mu <- mean(daily$MEAN_SENTIMENT, na.rm = TRUE)
  sigma <- sd(daily$MEAN_SENTIMENT, na.rm = TRUE)
  daily$FOMO_Z <- if (sigma > 0) (daily$MEAN_SENTIMENT - mu) / sigma else 0

  # Merge with regimes
  daily_regime <- daily %>%
    inner_join(
      regime_result$regime_assignments %>% select(DATE, REGIME_LABEL),
      by = "DATE"
    )

  # FOMO by regime
  fomo_by_regime <- daily_regime %>%
    group_by(REGIME = REGIME_LABEL) %>%
    summarise(
      MEAN_FOMO_Z   = mean(FOMO_Z, na.rm = TRUE),
      MEDIAN_FOMO_Z = median(FOMO_Z, na.rm = TRUE),
      STD_FOMO_Z    = sd(FOMO_Z, na.rm = TRUE),
      PCT_EUPHORIA  = mean(FOMO_Z > 1.5, na.rm = TRUE),
      PCT_PANIC     = mean(FOMO_Z < -1.5, na.rm = TRUE),
      N_DAYS        = n(),
      .groups       = "drop"
    )

  list(
    sentiment_daily = daily_regime,
    fomo_by_regime  = fomo_by_regime,
    n_articles      = nrow(news_data),
    n_scored        = sum(!is.na(news_data[[sent_col]]))
  )
}
