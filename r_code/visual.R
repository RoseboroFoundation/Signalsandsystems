# =============================================================================
# Visualization Module for Signals & Systems Dissertation (R Version)
#
# Generates publication-quality ggplot2 figures across three essays:
#   Essay 1 (33): Volatility regimes & Fama-French five-factor model
#   Essay 2 (33): Culture war event study with regime conditioning
#   Essay 3 (20): Informed insider trading around political decisions
#
# Color scheme: navy (#1B2A4A) + gold (#C5A55A) professional theme
#
# Usage:
#   source("database.R")
#   source("visual.R")
#   store <- load_result_store()
#   plots <- generate_all_charts(store)
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)
library(stringr)
library(scales)
library(purrr)
library(glue)

# =============================================================================
# Theme & Color Palettes
# =============================================================================

NAVY <- "#1B2A4A"
GOLD <- "#C5A55A"

theme_dissertation <- function(base_size = 11) {
  theme_minimal(base_size = base_size, base_family = "serif") +
    theme(
      plot.title       = element_text(size = base_size + 2, face = "bold", color = NAVY),
      plot.subtitle    = element_text(size = base_size, color = "grey40"),
      axis.title       = element_text(size = base_size, color = NAVY),
      axis.text        = element_text(size = base_size - 2),
      legend.title     = element_text(size = base_size - 1, face = "bold"),
      legend.text      = element_text(size = base_size - 2),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      strip.text       = element_text(face = "bold", size = base_size - 1),
      plot.margin      = margin(10, 10, 10, 10)
    )
}

# Regime colors
C_REGIME <- c(Low = "#27ae60", Medium = "#f39c12", High = "#e74c3c",
              `Low VIX` = "#27ae60", `Medium VIX` = "#f39c12", `High VIX` = "#e74c3c")
scale_fill_regime <- function(...) {
  scale_fill_manual(values = C_REGIME, ...)
}
scale_color_regime <- function(...) {
  scale_color_manual(values = C_REGIME, ...)
}

# Political leaning colors
C_LEAN <- c(Conservative = "#c0392b", Liberal = "#2980b9", Mixed = "#7f8c8d")
scale_fill_lean <- function(...) {
  scale_fill_manual(values = C_LEAN, ...)
}

# Treatment / Control colors
C_TC <- c(Treatment = "#e74c3c", Control = "#3498db")

# Political / Control colors (Essay 3)
C_POL_CTRL <- c(POLITICAL = "#e74c3c", CONTROL = "#3498db")

# Factor colors
C_FACTORS <- c(MKT_RF = "#2c3e50", SMB = "#e67e22", HML = "#27ae60",
               RMW = "#8e44ad", CMA = "#2980b9")

# Sequential palette
C_SEQ <- c("#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e84393", "#00b894", "#fdcb6e", "#6c5ce7")

# Proximity palette (Essay 3)
C_PROXIMITY <- c("#264653", "#2a9d8f", "#e9c46a", "#f4a261")


# =============================================================================
# Helpers
# =============================================================================

sig_stars <- function(p) {
  case_when(
    is.na(p)   ~ "",
    p < 0.001  ~ "***",
    p < 0.01   ~ "**",
    p < 0.05   ~ "*",
    p < 0.10   ~ "\u2020",
    TRUE       ~ ""
  )
}

safe_float <- function(x, default = NA_real_) {
  result <- suppressWarnings(as.numeric(x))
  ifelse(is.na(result), default, result)
}

first_col <- function(df, candidates) {
  for (c in candidates) {
    if (c %in% names(df)) return(c)
  }
  NULL
}

empty_plot <- function(title = "No data available") {
  ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = title,
             size = 5, color = "grey60") +
    theme_void() +
    labs(title = title)
}

FIGURE_DIR <- file.path(tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "."), "..", "figures")


# =============================================================================
# Essay 1 — Volatility Regimes & Fama-French Five-Factor (33 charts)
# =============================================================================

#' E1-01: FF5 factor premia by regime (grouped bar)
plot_factor_premia_by_regime <- function(store) {
  df <- store$e1_factor_premia
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E1-01: No factor premia data"))

  factors <- c("MKT_RF", "SMB", "HML", "RMW", "CMA")
  ann_cols <- paste0(factors, "_MEAN_ANN")
  present <- intersect(ann_cols, names(df))
  if (length(present) == 0) return(empty_plot("E1-01: No annualised columns"))

  plot_df <- df |>
    select(REGIME, all_of(present)) |>
    pivot_longer(-REGIME, names_to = "factor", values_to = "value") |>
    mutate(
      factor = str_remove(factor, "_MEAN_ANN"),
      value  = safe_float(value)
    )

  ggplot(plot_df, aes(x = factor, y = value, fill = REGIME)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "white", linewidth = 0.3) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    scale_fill_regime() +
    scale_y_continuous(labels = percent_format()) +
    labs(title = "FF5 Factor Premia by VIX Regime",
         y = "Annualised Mean Return", x = NULL, fill = "Regime") +
    theme_dissertation()
}


#' E1-02: Alpha by regime (bar)
plot_alpha_by_regime <- function(store) {
  df <- store$e1_ff5_coefficients
  if (is.null(df) || nrow(df) == 0 || !"ALPHA" %in% names(df))
    return(empty_plot("E1-02: No FF5 coefficients"))

  df <- df |> mutate(ALPHA = safe_float(ALPHA), ALPHA_P = safe_float(ALPHA_P))

  ggplot(df, aes(x = REGIME, y = ALPHA, fill = REGIME)) +
    geom_col(width = 0.5, color = "white", linewidth = 0.3) +
    geom_text(aes(label = paste0(round(ALPHA, 4), sig_stars(ALPHA_P))),
              vjust = ifelse(df$ALPHA >= 0, -0.3, 1.3), size = 3.5) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    scale_fill_regime() +
    labs(title = "FF5 Intercept (Alpha) by VIX Regime",
         y = "Alpha (daily)", x = NULL) +
    theme_dissertation() +
    theme(legend.position = "none")
}


#' E1-03: Beta heatmap
plot_beta_heatmap <- function(store) {
  df <- store$e1_ff5_coefficients
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E1-03: No FF5 coefficients"))
  beta_cols <- names(df)[str_ends(names(df), "_BETA")]
  if (length(beta_cols) == 0) return(empty_plot("E1-03: No beta columns"))

  plot_df <- df |>
    select(REGIME, all_of(beta_cols)) |>
    pivot_longer(-REGIME, names_to = "factor", values_to = "beta") |>
    mutate(factor = str_remove(factor, "_BETA"), beta = safe_float(beta))

  ggplot(plot_df, aes(x = factor, y = REGIME, fill = beta)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.3f", beta)), size = 3.5) +
    scale_fill_gradient2(low = "#2980b9", mid = "white", high = "#e74c3c",
                         midpoint = 0, name = "Beta") +
    labs(title = "FF5 Factor Loadings by VIX Regime", x = NULL, y = NULL) +
    theme_dissertation()
}


#' E1-04: t-statistic heatmap
plot_tstat_heatmap <- function(store) {
  df <- store$e1_ff5_coefficients
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E1-04: No FF5 coefficients"))
  t_cols <- names(df)[str_ends(names(df), "_T")]
  if (length(t_cols) == 0) return(empty_plot("E1-04: No t-stat columns"))

  plot_df <- df |>
    select(REGIME, all_of(t_cols)) |>
    pivot_longer(-REGIME, names_to = "factor", values_to = "tstat") |>
    mutate(factor = str_remove(factor, "_T"), tstat = safe_float(tstat))

  ggplot(plot_df, aes(x = factor, y = REGIME, fill = tstat)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.2f", tstat)), size = 3.5) +
    scale_fill_gradient2(low = "#2980b9", mid = "white", high = "#e74c3c",
                         midpoint = 0, limits = c(-4, 4), oob = squish,
                         name = "t-statistic") +
    labs(title = "FF5 Factor Loading t-Statistics by Regime", x = NULL, y = NULL) +
    theme_dissertation()
}


#' E1-05: Chow structural break test
plot_chow_test <- function(store) {
  df <- store$e1_chow_test
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E1-05: No Chow test"))
  row <- df[1, ]
  f_stat <- safe_float(row$F_STAT %||% row$f_stat)
  p_val  <- safe_float(row$P_VALUE %||% row$p_value)
  df_num <- safe_float(row$DF_NUMERATOR %||% row$df_numerator)
  df_den <- safe_float(row$DF_DENOMINATOR %||% row$df_denominator)

  if (is.na(df_num) || is.na(df_den)) return(empty_plot("E1-05: Missing df"))

  x_vals <- seq(0, max(f_stat * 1.5, 5), length.out = 500)
  f_dens <- df(x_vals, df_num, df_den)
  f_crit <- qf(0.95, df_num, df_den)
  sig <- ifelse(p_val < 0.05, "YES", "NO")

  plot_df <- tibble(x = x_vals, density = f_dens)

  ggplot(plot_df, aes(x, density)) +
    geom_area(fill = "#3498db", alpha = 0.2) +
    geom_line(color = "#3498db", linewidth = 1) +
    geom_vline(xintercept = f_crit, linetype = "dashed", color = "#e67e22", linewidth = 0.8) +
    geom_vline(xintercept = f_stat, color = "#e74c3c", linewidth = 1.5) +
    annotate("text", x = f_crit, y = max(f_dens) * 0.9,
             label = sprintf("Critical (5%%) = %.2f", f_crit),
             hjust = -0.1, size = 3, color = "#e67e22") +
    labs(title = sprintf("Chow Structural Break Test\nF = %.2f, p = %.4f — Break: %s",
                         f_stat, p_val, sig),
         x = "F statistic", y = "Density") +
    theme_dissertation()
}


#' E1-06: R-squared by regime
plot_rsquared_by_regime <- function(store) {
  df <- store$e1_ff5_coefficients
  if (is.null(df) || nrow(df) == 0 || !"R_SQUARED" %in% names(df))
    return(empty_plot("E1-06: No R-squared"))

  df <- df |> mutate(R_SQUARED = safe_float(R_SQUARED))

  ggplot(df, aes(x = REGIME, y = R_SQUARED, fill = REGIME)) +
    geom_col(width = 0.5, color = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", R_SQUARED)), vjust = -0.3, size = 3.5) +
    scale_fill_regime() +
    coord_cartesian(ylim = c(0, min(max(df$R_SQUARED, na.rm = TRUE) * 1.3, 1))) +
    labs(title = expression("FF5 Model Fit (R"^2*") by VIX Regime"),
         y = expression(R^2), x = NULL) +
    theme_dissertation() +
    theme(legend.position = "none")
}


#' E1-08: CW stock alpha boxplot
plot_cw_alpha_boxplot <- function(store) {
  df <- store$e1_cw_stock
  if (is.null(df) || nrow(df) == 0 || !"ALPHA" %in% names(df))
    return(empty_plot("E1-08: No CW stock results"))

  df <- df |> mutate(ALPHA = safe_float(ALPHA))

  ggplot(df, aes(x = REGIME, y = ALPHA, fill = REGIME)) +
    geom_boxplot(outlier.size = 1, color = NAVY) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
    scale_fill_regime() +
    labs(title = "Culture War Stock Alphas by VIX Regime",
         y = "Alpha (daily)", x = "VIX Regime") +
    theme_dissertation() +
    theme(legend.position = "none")
}


#' E1-15: FOMO z-scores by regime
plot_fomo_by_regime <- function(store) {
  df <- store$e1_fomo
  if (is.null(df) || nrow(df) == 0 || !"MEAN_FOMO_Z" %in% names(df))
    return(empty_plot("E1-15: No FOMO data"))

  df <- df |> mutate(
    MEAN_FOMO_Z = safe_float(MEAN_FOMO_Z),
    STD_FOMO_Z  = safe_float(STD_FOMO_Z)
  )

  ggplot(df, aes(x = REGIME, y = MEAN_FOMO_Z, fill = REGIME)) +
    geom_col(width = 0.5, color = "white", linewidth = 0.3) +
    geom_errorbar(aes(ymin = MEAN_FOMO_Z - STD_FOMO_Z,
                      ymax = MEAN_FOMO_Z + STD_FOMO_Z),
                  width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
    scale_fill_regime() +
    labs(title = "Fear-of-Missing-Out Z-Scores by VIX Regime",
         y = "Mean FOMO Z-Score", x = NULL) +
    theme_dissertation() +
    theme(legend.position = "none")
}


# --- Essay 1 Matched Control Plots ---

#' E1-19: Matched t-test forest plot
plot_matched_ttest_forest <- function(store) {
  df <- store$e1_matched_ttest
  if (is.null(df) || nrow(df) == 0 || !"VARIABLE" %in% names(df))
    return(empty_plot("E1-19: No matched t-test results"))

  df <- df |> mutate(
    MEAN_DELTA = safe_float(MEAN_DELTA),
    STD_DELTA  = safe_float(STD_DELTA %||% 0),
    P_VALUE    = safe_float(P_VALUE),
    label = if ("REGIME" %in% names(df)) paste0(VARIABLE, " (", REGIME, ")") else VARIABLE,
    sig_group  = case_when(
      !is.na(BH_SIGNIFICANT) & BH_SIGNIFICANT ~ "BH significant",
      P_VALUE < 0.05 ~ "p < 0.05",
      TRUE ~ "Not significant"
    )
  )

  ggplot(df, aes(y = reorder(label, MEAN_DELTA), x = MEAN_DELTA, fill = sig_group)) +
    geom_col(width = 0.6, color = "white", linewidth = 0.2) +
    geom_errorbarh(aes(xmin = MEAN_DELTA - STD_DELTA,
                       xmax = MEAN_DELTA + STD_DELTA),
                   height = 0.2) +
    geom_vline(xintercept = 0, linewidth = 0.4) +
    scale_fill_manual(values = c("BH significant" = "#e74c3c",
                                  "p < 0.05"       = "#f39c12",
                                  "Not significant" = "#bdc3c7")) +
    labs(title = "Matched Control: Treatment-Control Differences",
         x = "Mean Delta (Treatment - Control)", y = NULL, fill = "Significance") +
    theme_dissertation()
}


#' E1-21: Regime amplification
plot_matched_amplification <- function(store) {
  df <- store$e1_matched_amp
  if (is.null(df) || nrow(df) == 0 || !"VARIABLE" %in% names(df))
    return(empty_plot("E1-21: No amplification data"))

  df <- df |> mutate(
    MEAN_DELTA_LOW  = safe_float(MEAN_DELTA_LOW),
    MEAN_DELTA_HIGH = safe_float(MEAN_DELTA_HIGH),
    P_VALUE         = safe_float(P_VALUE)
  )

  plot_df <- df |>
    select(VARIABLE, MEAN_DELTA_LOW, MEAN_DELTA_HIGH) |>
    pivot_longer(-VARIABLE, names_to = "regime", values_to = "delta") |>
    mutate(regime = ifelse(regime == "MEAN_DELTA_LOW", "Low VIX", "High VIX"))

  ggplot(plot_df, aes(x = VARIABLE, y = delta, fill = regime)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, color = "white", linewidth = 0.2) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    scale_fill_manual(values = c("Low VIX" = "#27ae60", "High VIX" = "#e74c3c")) +
    labs(title = "Regime Amplification: High vs Low VIX",
         y = "Mean Delta", x = NULL, fill = "Regime") +
    theme_dissertation() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}


# =============================================================================
# Essay 2 — Culture War Event Study & DiD (33 charts)
# =============================================================================

#' E2-02: CAR by political leaning
plot_car_by_leaning <- function(store) {
  df <- store$e2_car_panel
  if (is.null(df) || nrow(df) == 0 || !"LEAN" %in% names(df))
    return(empty_plot("E2-02: No CAR/leaning data"))

  grp <- df |>
    mutate(CAR_POST = safe_float(CAR_POST)) |>
    group_by(LEAN) |>
    summarise(mean = mean(CAR_POST, na.rm = TRUE),
              sd   = sd(CAR_POST, na.rm = TRUE),
              n    = n(), .groups = "drop") |>
    mutate(ci = 1.96 * sd / sqrt(n))

  ggplot(grp, aes(x = LEAN, y = mean, fill = LEAN)) +
    geom_col(width = 0.5, color = "white", linewidth = 0.3) +
    geom_errorbar(aes(ymin = mean - ci, ymax = mean + ci), width = 0.2) +
    geom_text(aes(label = sprintf("%.3f\n(N=%d)", mean, n)),
              vjust = -0.5, size = 3) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    scale_fill_lean() +
    labs(title = "Culture War Event Impact by Political Leaning",
         y = "Mean CAR (Post-Event)", x = NULL) +
    theme_dissertation() +
    theme(legend.position = "none")
}


#' E2-06: DiD coefficient forest plot
plot_did_coefficients <- function(store) {
  df <- store$e2_did_coeff
  if (is.null(df) || nrow(df) == 0 || !"VARIABLE" %in% names(df))
    return(empty_plot("E2-06: No DiD coefficients"))

  df <- df |> mutate(
    COEFFICIENT = safe_float(COEFFICIENT),
    STD_ERROR   = safe_float(STD_ERROR %||% 0),
    P_VALUE     = safe_float(P_VALUE),
    label       = if ("SPECIFICATION" %in% names(df))
                    paste0(VARIABLE, " (", SPECIFICATION, ")") else VARIABLE,
    sig         = P_VALUE < 0.05
  )

  ggplot(df, aes(y = reorder(label, COEFFICIENT), x = COEFFICIENT,
                  fill = sig)) +
    geom_col(width = 0.5, color = "white", linewidth = 0.2) +
    geom_errorbarh(aes(xmin = COEFFICIENT - 1.96 * STD_ERROR,
                       xmax = COEFFICIENT + 1.96 * STD_ERROR),
                   height = 0.2) +
    geom_vline(xintercept = 0, linewidth = 0.4) +
    scale_fill_manual(values = c(`TRUE` = "#e74c3c", `FALSE` = "#bdc3c7"),
                      labels = c("p >= 0.05", "p < 0.05"), name = "Significance") +
    labs(title = "Difference-in-Differences Regression Coefficients",
         x = "Coefficient", y = NULL) +
    theme_dissertation()
}


#' E2-07: Parallel trends
plot_parallel_trends <- function(store) {
  df <- store$e2_parallel
  if (is.null(df) || nrow(df) == 0 || !"DAY" %in% names(df))
    return(empty_plot("E2-07: No parallel trends data"))

  df <- df |>
    arrange(DAY) |>
    mutate(
      DAY         = as.integer(DAY),
      COEFFICIENT = safe_float(COEFFICIENT),
      STD_ERROR   = safe_float(STD_ERROR %||% 0),
      ci_lo       = COEFFICIENT - 1.96 * STD_ERROR,
      ci_hi       = COEFFICIENT + 1.96 * STD_ERROR
    )

  subtitle <- ""
  if ("JOINT_F_STAT" %in% names(df)) {
    f_stat <- safe_float(df$JOINT_F_STAT[1])
    f_p    <- safe_float(df$JOINT_P_VALUE[1])
    passes <- if ("PASSES" %in% names(df)) df$PASSES[1] else NA
    subtitle <- sprintf("Joint F = %.2f, p = %.4f", f_stat, f_p)
    if (!is.na(passes)) subtitle <- paste0(subtitle, " -- Passes: ", ifelse(passes, "YES", "NO"))
  }

  ggplot(df, aes(x = DAY, y = COEFFICIENT)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), fill = "#3498db", alpha = 0.2) +
    geom_line(color = "#2980b9", linewidth = 1) +
    geom_point(color = "#2980b9", size = 2) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red", alpha = 0.5) +
    labs(title = "Parallel Trends Test",
         subtitle = subtitle,
         x = "Days Relative to Event",
         y = expression("Treatment" %*% "Day Coefficient")) +
    theme_dissertation()
}


#' E2-04: CAR by regime (boxplot)
plot_car_by_regime <- function(store) {
  df <- store$e2_car_panel
  if (is.null(df) || nrow(df) == 0 || !"REGIME" %in% names(df))
    return(empty_plot("E2-04: No CAR/regime data"))

  df <- df |> mutate(CAR_POST = safe_float(CAR_POST))

  ggplot(df, aes(x = REGIME, y = CAR_POST, fill = REGIME)) +
    geom_boxplot(outlier.size = 1, color = NAVY) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.5) +
    scale_fill_regime() +
    labs(title = "Post-Event CAR by VIX Regime",
         y = "CAR Post-Event", x = NULL) +
    theme_dissertation() +
    theme(legend.position = "none")
}


#' E2-14: Political alignment distribution
plot_alignment_distribution <- function(store) {
  df <- store$e2_alignment
  if (is.null(df) || nrow(df) == 0 || !"ALIGNMENT_SCORE" %in% names(df))
    return(empty_plot("E2-14: No alignment data"))

  df <- df |> mutate(ALIGNMENT_SCORE = safe_float(ALIGNMENT_SCORE))
  m <- mean(df$ALIGNMENT_SCORE, na.rm = TRUE)

  ggplot(df, aes(x = ALIGNMENT_SCORE)) +
    geom_histogram(bins = 30, fill = "#9b59b6", color = "white", alpha = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
    geom_vline(xintercept = m, color = "#e74c3c", linewidth = 1) +
    annotate("text", x = m, y = Inf, label = sprintf("Mean = %.3f", m),
             vjust = 2, hjust = -0.1, size = 3.5, color = "#e74c3c") +
    labs(title = "Political Alignment Score Distribution",
         x = "Alignment Score (- = Liberal, + = Conservative)", y = "Count") +
    theme_dissertation()
}


#' E2-18: Multi-window CAR by horizon
plot_multiwindow_by_horizon <- function(store) {
  df <- store$e2_mw_summary
  if (is.null(df) || nrow(df) == 0 || !"WINDOW" %in% names(df))
    return(empty_plot("E2-18: No multi-window summary"))

  df <- df |> mutate(
    MEAN_CAR_TREAT = safe_float(MEAN_CAR_TREAT),
    P_VALUE_VS_ZERO = safe_float(P_VALUE_VS_ZERO),
    sig = P_VALUE_VS_ZERO < 0.05
  )

  if ("WINDOW_DAYS" %in% names(df)) df <- df |> arrange(WINDOW_DAYS)

  ggplot(df, aes(x = reorder(WINDOW, seq_len(nrow(df))),
                  y = MEAN_CAR_TREAT, fill = sig)) +
    geom_col(width = 0.6, color = "white", linewidth = 0.2) +
    geom_text(aes(label = sig_stars(P_VALUE_VS_ZERO)),
              vjust = -0.3, size = 4) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
    scale_fill_manual(values = c(`TRUE` = "#e74c3c", `FALSE` = "#bdc3c7"),
                      guide = "none") +
    labs(title = "Cumulative Abnormal Returns Across Event Windows",
         y = "Mean CAR (Treatment)", x = NULL) +
    theme_dissertation() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}


#' E2-21: Contagion summary
plot_contagion_summary <- function(store) {
  df <- store$e2_cont_summary
  if (is.null(df) || nrow(df) == 0 || !"WINDOW" %in% names(df))
    return(empty_plot("E2-21: No contagion summary"))

  df <- df |> mutate(
    MEAN_PEER_CAR = safe_float(MEAN_PEER_CAR),
    P_VALUE_VS_ZERO = safe_float(P_VALUE_VS_ZERO),
    sig = P_VALUE_VS_ZERO < 0.05
  )

  if ("WINDOW_DAYS" %in% names(df)) df <- df |> arrange(WINDOW_DAYS)

  ggplot(df, aes(x = reorder(WINDOW, seq_len(nrow(df))),
                  y = MEAN_PEER_CAR, fill = sig)) +
    geom_col(width = 0.6, color = "white", linewidth = 0.2) +
    geom_text(aes(label = sig_stars(P_VALUE_VS_ZERO)),
              vjust = -0.3, size = 4) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
    scale_fill_manual(values = c(`TRUE` = "#e67e22", `FALSE` = "#bdc3c7"),
                      guide = "none") +
    labs(title = "Contagion: Peer Firm Abnormal Returns",
         y = "Mean Peer CAR", x = NULL) +
    theme_dissertation() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}


# =============================================================================
# Essay 3 — Informed Insider Trading Around Political Decisions (20 charts)
# =============================================================================

#' E3-01: Directional accuracy by sample
plot_directional_accuracy <- function(store) {
  df <- store$e3_informed_trading
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E3-01: No informed trading data"))

  main <- df |>
    filter(CUT %in% c("ALL", "SELLS_ONLY", "BUYS_ONLY")) |>
    mutate(
      METRIC_VALUE = safe_float(METRIC_VALUE),
      METRIC_PVAL  = safe_float(METRIC_PVAL),
      label        = paste0(sprintf("%.1f%%", METRIC_VALUE * 100),
                            " ", sig_stars(METRIC_PVAL))
    )

  if (nrow(main) == 0) main <- df |> head(10) |>
    mutate(METRIC_VALUE = safe_float(METRIC_VALUE),
           METRIC_PVAL  = safe_float(METRIC_PVAL))

  ggplot(main, aes(y = CUT, x = METRIC_VALUE, fill = SAMPLE)) +
    geom_col(width = 0.6, color = "white", linewidth = 0.2) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey50") +
    geom_text(aes(label = label), hjust = -0.1, size = 3) +
    scale_fill_manual(values = C_POL_CTRL) +
    scale_x_continuous(labels = percent_format()) +
    labs(title = "Informed Trading: Directional Accuracy\n(Political vs Matched Control)",
         x = "Directional Accuracy", y = NULL, fill = "Sample") +
    theme_dissertation()
}


#' E3-03: Proximity accuracy (proximity gradient)
plot_proximity_gradient <- function(store) {
  df <- store$e3_informed_proximity
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E3-03: No proximity data"))

  pol <- df |>
    filter(SAMPLE == "POLITICAL") |>
    mutate(METRIC_VALUE = safe_float(METRIC_VALUE),
           METRIC_PVAL  = safe_float(METRIC_PVAL))

  sells <- pol |> filter(str_ends(CUT, "_SELLS"))
  if (nrow(sells) == 0) sells <- pol

  ctrl_line <- df |>
    filter(CUT == "CONTROL_ALL_SELLS") |>
    pull(METRIC_VALUE) |>
    safe_float()

  p <- ggplot(sells, aes(x = reorder(CUT, seq_len(nrow(sells))),
                           y = METRIC_VALUE)) +
    geom_col(fill = C_PROXIMITY[seq_len(min(nrow(sells), length(C_PROXIMITY)))],
             width = 0.7, color = "black", linewidth = 0.3) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
    geom_text(aes(label = paste0(sprintf("%.1f%%", METRIC_VALUE * 100),
                                  " ", sig_stars(METRIC_PVAL))),
              vjust = -0.3, size = 3) +
    scale_y_continuous(labels = percent_format()) +
    labs(title = "Sells Accuracy by Proximity to Event\n(Political vs Control Baseline)",
         y = "Directional Accuracy", x = NULL) +
    theme_dissertation() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  if (!is.na(ctrl_line)) {
    p <- p + geom_hline(yintercept = ctrl_line, linetype = "dotted",
                         color = "#3498db", linewidth = 1)
  }
  p
}


#' E3-12: TOST equivalence test
plot_tost_equivalence <- function(store) {
  df <- store$e3_tost
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E3-12: No TOST data"))

  df <- df |> mutate(
    DELTA  = safe_float(DELTA_RAW %||% DELTA %||% 0.2),
    MEAN   = safe_float(MEAN %||% 0),
    SE     = safe_float(SE %||% 0),
    ci_lo  = MEAN - 1.645 * SE,
    ci_hi  = MEAN + 1.645 * SE,
    EQUIVALENT = safe_float(EQUIVALENT %||% 0),
    P_TOST = safe_float(P_TOST),
    idx    = row_number(),
    label  = if ("EVENT_CATEGORY" %in% names(df)) EVENT_CATEGORY else
               if ("MARGIN_NAME" %in% names(df)) MARGIN_NAME else as.character(idx),
    color  = ifelse(EQUIVALENT == 1, "#27ae60", "#e74c3c")
  )

  ggplot(df, aes(y = reorder(label, idx))) +
    geom_segment(aes(x = ci_lo, xend = ci_hi, color = color),
                 linewidth = 2.5, lineend = "round") +
    geom_point(aes(x = MEAN, color = color), size = 3) +
    geom_vline(xintercept = 0, linewidth = 0.5) +
    geom_text(aes(x = ci_hi, label = sprintf("p=%.3f", P_TOST)),
              hjust = -0.1, size = 3) +
    scale_color_identity() +
    labs(title = "TOST Equivalence Tests (90% CI vs SESOI)",
         x = "Mean Abnormal Net Trading", y = NULL) +
    theme_dissertation()
}


#' E3-13: Placebo distribution
plot_placebo_distribution <- function(store) {
  df <- store$e3_placebo
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E3-13: No placebo data"))

  row1 <- df[1, ]
  obs  <- safe_float(row1$OBSERVED_DIFF %||% row1$OBSERVED_STAT %||% 0)
  mu   <- safe_float(row1$NULL_MEAN %||% row1$PLACEBO_MEAN %||% 0)
  sigma <- safe_float(row1$NULL_STD %||% row1$PLACEBO_STD %||% 1)
  p_val <- safe_float(row1$P_VALUE %||% row1$EMPIRICAL_P %||% 1)

  x_range <- seq(mu - 4 * sigma, mu + 4 * sigma, length.out = 200)
  dens    <- dnorm(x_range, mu, sigma)
  norm_df <- tibble(x = x_range, y = dens)

  ggplot(norm_df, aes(x, y)) +
    geom_area(fill = "#3498db", alpha = 0.3) +
    geom_line(color = "#3498db") +
    geom_vline(xintercept = obs, color = "#e74c3c", linewidth = 1.5) +
    geom_vline(xintercept = mu, linetype = "dashed", color = "#3498db") +
    annotate("text", x = obs, y = max(dens) * 0.8,
             label = sprintf("Observed (%.4f)", obs),
             hjust = -0.1, size = 3, color = "#e74c3c") +
    labs(title = sprintf("Placebo Test (p = %.4f)", p_val),
         x = "Test Statistic", y = "Density") +
    theme_dissertation()
}


#' E3-06: Trade-size decile accuracy
plot_size_accuracy <- function(store) {
  df <- store$e3_size_accuracy
  if (is.null(df) || nrow(df) == 0) return(empty_plot("E3-06: No size-accuracy data"))

  df <- df |> mutate(
    SIZE_DECILE  = safe_float(SIZE_DECILE),
    PCT_ACCURATE = safe_float(PCT_ACCURATE)
  )

  ggplot(df, aes(x = SIZE_DECILE, y = PCT_ACCURATE, color = SAMPLE)) +
    geom_line(linewidth = 1.5) +
    geom_point(size = 3) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
    scale_color_manual(values = C_POL_CTRL) +
    scale_y_continuous(labels = percent_format()) +
    labs(title = "Accuracy by Trade-Size Decile\n(Political vs Control)",
         x = "Trade-Size Decile", y = "Directional Accuracy",
         color = "Sample") +
    theme_dissertation()
}


# =============================================================================
# Transition Matrix Heatmap (Essay 1)
# =============================================================================

#' Plot Markov transition matrix heatmap
plot_transition_matrix <- function(trans_matrix) {
  if (is.null(trans_matrix) || nrow(trans_matrix) == 0)
    return(empty_plot("No transition matrix data"))

  if (is.matrix(trans_matrix)) {
    trans_matrix <- as.data.frame(trans_matrix) |>
      rownames_to_column("from")
  }

  plot_df <- trans_matrix |>
    pivot_longer(-1, names_to = "to", values_to = "prob") |>
    mutate(prob = safe_float(prob))
  names(plot_df)[1] <- "from"

  ggplot(plot_df, aes(x = to, y = from, fill = prob)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.3f", prob)), size = 4) +
    scale_fill_gradient(low = "white", high = GOLD, name = "Probability") +
    labs(title = "Markov Regime Transition Probabilities",
         x = "To Regime", y = "From Regime") +
    theme_dissertation() +
    coord_fixed()
}


# =============================================================================
# Regime Summary Dashboard
# =============================================================================

#' Multi-panel regime summary (requires patchwork)
plot_regime_summary <- function(store) {
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("Install 'patchwork' for multi-panel layouts")
    return(plot_alpha_by_regime(store))
  }

  p1 <- plot_alpha_by_regime(store)
  p2 <- plot_fomo_by_regime(store)
  p3 <- plot_rsquared_by_regime(store)
  p4 <- plot_cw_alpha_boxplot(store)

  patchwork::wrap_plots(p1, p2, p3, p4, ncol = 2) +
    patchwork::plot_annotation(
      title    = "Essay 1 -- Volatility Regimes & FF5 Summary",
      theme    = theme(plot.title = element_text(size = 15, face = "bold", color = NAVY))
    )
}


# =============================================================================
# Chart Registry & Generation
# =============================================================================

ESSAY1_CHARTS <- list(
  list(name = "e1_01_factor_premia_by_regime",  func = plot_factor_premia_by_regime, desc = "FF5 factor premia by regime"),
  list(name = "e1_02_alpha_by_regime",          func = plot_alpha_by_regime,         desc = "FF5 alpha by regime"),
  list(name = "e1_03_beta_heatmap",             func = plot_beta_heatmap,            desc = "FF5 beta heatmap"),
  list(name = "e1_04_tstat_heatmap",            func = plot_tstat_heatmap,           desc = "FF5 t-statistic heatmap"),
  list(name = "e1_05_chow_test",                func = plot_chow_test,               desc = "Chow structural break test"),
  list(name = "e1_06_rsquared_by_regime",       func = plot_rsquared_by_regime,      desc = "R-squared by regime"),
  list(name = "e1_08_cw_alpha_boxplot",         func = plot_cw_alpha_boxplot,        desc = "CW stock alpha boxplot"),
  list(name = "e1_15_fomo_by_regime",           func = plot_fomo_by_regime,          desc = "FOMO z-scores by regime"),
  list(name = "e1_19_matched_ttest_forest",     func = plot_matched_ttest_forest,    desc = "Matched t-test forest plot"),
  list(name = "e1_21_matched_amplification",    func = plot_matched_amplification,   desc = "Regime amplification"),
  list(name = "e1_33_regime_summary",           func = plot_regime_summary,          desc = "Essay 1 summary dashboard")
)

ESSAY2_CHARTS <- list(
  list(name = "e2_02_car_by_leaning",           func = plot_car_by_leaning,          desc = "CAR by political leaning"),
  list(name = "e2_04_car_by_regime",            func = plot_car_by_regime,           desc = "CAR by VIX regime"),
  list(name = "e2_06_did_coefficients",         func = plot_did_coefficients,        desc = "DiD coefficient forest plot"),
  list(name = "e2_07_parallel_trends",          func = plot_parallel_trends,         desc = "Parallel trends test"),
  list(name = "e2_14_alignment_distribution",   func = plot_alignment_distribution,  desc = "Political alignment distribution"),
  list(name = "e2_18_multiwindow_by_horizon",   func = plot_multiwindow_by_horizon,  desc = "Multi-window CAR by horizon"),
  list(name = "e2_21_contagion_summary",        func = plot_contagion_summary,       desc = "Contagion peer CARs")
)

ESSAY3_CHARTS <- list(
  list(name = "e3_01_directional_accuracy",     func = plot_directional_accuracy,    desc = "Directional accuracy by sample"),
  list(name = "e3_03_proximity_accuracy",       func = plot_proximity_gradient,      desc = "Accuracy by proximity to event"),
  list(name = "e3_06_size_accuracy",            func = plot_size_accuracy,           desc = "Trade-size decile accuracy"),
  list(name = "e3_12_tost_equivalence",         func = plot_tost_equivalence,        desc = "TOST equivalence tests"),
  list(name = "e3_13_placebo_test",             func = plot_placebo_distribution,    desc = "Placebo permutation test")
)

ALL_CHARTS <- c(ESSAY1_CHARTS, ESSAY2_CHARTS, ESSAY3_CHARTS)


#' Generate and save all charts
#'
#' @param store Named list from load_result_store()
#' @param chart_list List of chart specs (default: ALL_CHARTS)
#' @param save Logical; save PNG files to figures/ directory?
#' @param width Plot width in inches (default: 10)
#' @param height Plot height in inches (default: 7)
#' @param dpi Resolution (default: 300)
#' @return A tibble of results
generate_all_charts <- function(store, chart_list = ALL_CHARTS,
                                save = TRUE, width = 10, height = 7, dpi = 300) {
  if (save) dir.create(FIGURE_DIR, showWarnings = FALSE, recursive = TRUE)

  results <- map_dfr(chart_list, function(spec) {
    t0 <- proc.time()
    status <- "SUCCESS"
    error_msg <- NA_character_

    tryCatch({
      p <- spec$func(store)

      if (save) {
        out_path <- file.path(FIGURE_DIR, paste0(spec$name, ".png"))
        ggsave(out_path, p, width = width, height = height, dpi = dpi, bg = "white")
      }
    }, error = function(e) {
      status    <<- "FAILED"
      error_msg <<- conditionMessage(e)
    })

    elapsed <- (proc.time() - t0)["elapsed"]
    message(sprintf("  %-45s %-8s (%.1fs)", spec$name, status, elapsed))

    tibble(
      name     = spec$name,
      desc     = spec$desc,
      status   = status,
      error    = error_msg,
      duration = as.numeric(elapsed)
    )
  })

  n_ok   <- sum(results$status == "SUCCESS")
  n_fail <- sum(results$status == "FAILED")
  message(sprintf("\nComplete: %d succeeded, %d failed", n_ok, n_fail))

  results
}
