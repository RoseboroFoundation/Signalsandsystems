# =============================================================================
# Reporting Module for Signals & Systems Dissertation (R Version)
#
# Summary statistics and formatted Excel workbook generation.
#
# Usage:
#   source("database.R")
#   source("reporting.R")
#   conn  <- connect_db()
#   store <- load_result_store(conn)
#   stats <- summary_statistics(store)
#   generate_workbook(store, "output/dissertation_results.xlsx")
#   close_db(conn)
# =============================================================================

library(dplyr)
library(tidyr)
library(tibble)
library(purrr)
library(stringr)
library(glue)
library(openxlsx)

# =============================================================================
# Summary Statistics
# =============================================================================

#' Generate descriptive statistics for all loaded datasets
#'
#' @param store Named list from load_result_store()
#' @return A named list of tibbles with summary statistics
summary_statistics <- function(store) {
  stats <- list()

  # --- Event Summary ---
  cw <- store$cw_companies
  if (!is.null(cw) && nrow(cw) > 0) {
    stats$events <- tibble(
      Total_Events      = nrow(cw),
      Unique_Tickers    = n_distinct(cw$TICKER),
      Liberal           = sum(cw$ESTIMATED_POLITICAL_LEANING == "Liberal", na.rm = TRUE),
      Conservative      = sum(cw$ESTIMATED_POLITICAL_LEANING == "Conservative", na.rm = TRUE),
      Mixed             = sum(cw$ESTIMATED_POLITICAL_LEANING == "Mixed", na.rm = TRUE),
      Industries        = n_distinct(cw$INDUSTRY)
    )
  }

  # --- VIX Summary ---
  vix <- store$vix_data
  vix_col_name <- if (!is.null(vix) && "VIX" %in% names(vix)) "VIX" else "CLOSE"
  if (!is.null(vix) && nrow(vix) > 0 && vix_col_name %in% names(vix)) {
    vix_vals <- as.numeric(vix[[vix_col_name]])
    stats$vix <- tibble(
      Observations = length(na.omit(vix_vals)),
      Mean         = round(mean(vix_vals, na.rm = TRUE), 2),
      Std_Dev      = round(sd(vix_vals, na.rm = TRUE), 2),
      Min          = round(min(vix_vals, na.rm = TRUE), 2),
      Median       = round(median(vix_vals, na.rm = TRUE), 2),
      Max          = round(max(vix_vals, na.rm = TRUE), 2)
    )
  }

  # --- Essay 1: FF5 Coefficients ---
  ff5 <- store$e1_ff5_coefficients
  if (!is.null(ff5) && nrow(ff5) > 0) {
    stats$essay1_coefficients <- ff5 |>
      mutate(across(where(is.character), ~ suppressWarnings(as.numeric(.x)))) |>
      select(where(is.numeric)) |>
      summarise(across(everything(), list(
        mean = ~ round(mean(.x, na.rm = TRUE), 4),
        sd   = ~ round(sd(.x, na.rm = TRUE), 4)
      )))
  }

  # --- Essay 1: Factor Premia ---
  fp <- store$e1_factor_premia
  if (!is.null(fp) && nrow(fp) > 0) {
    stats$factor_premia <- fp
  }

  # --- Essay 1: Chow Test ---
  chow <- store$e1_chow_test
  if (!is.null(chow) && nrow(chow) > 0) {
    stats$chow_test <- chow
  }

  # --- Essay 2: CAR Panel Summary ---
  car <- store$e2_car_panel
  if (!is.null(car) && nrow(car) > 0) {
    car_cols <- intersect(c("CAR_PRE", "CAR_POST", "CAR_FULL"), names(car))
    if (length(car_cols) > 0) {
      stats$essay2_car_summary <- car |>
        mutate(across(all_of(car_cols), ~ suppressWarnings(as.numeric(.x)))) |>
        summarise(
          N = n(),
          across(all_of(car_cols), list(
            mean   = ~ round(mean(.x, na.rm = TRUE), 4),
            median = ~ round(median(.x, na.rm = TRUE), 4),
            sd     = ~ round(sd(.x, na.rm = TRUE), 4)
          ))
        )
    }

    # By leaning
    if ("LEAN" %in% names(car) && "CAR_POST" %in% names(car)) {
      stats$essay2_car_by_lean <- car |>
        mutate(CAR_POST = suppressWarnings(as.numeric(CAR_POST))) |>
        group_by(LEAN) |>
        summarise(
          N      = n(),
          Mean   = round(mean(CAR_POST, na.rm = TRUE), 4),
          Median = round(median(CAR_POST, na.rm = TRUE), 4),
          Std    = round(sd(CAR_POST, na.rm = TRUE), 4),
          .groups = "drop"
        )
    }
  }

  # --- Essay 2: DiD Coefficients ---
  did <- store$e2_did_coeff
  if (!is.null(did) && nrow(did) > 0) {
    stats$essay2_did <- did |>
      mutate(
        COEFFICIENT = safe_float(COEFFICIENT),
        STD_ERROR   = safe_float(STD_ERROR),
        P_VALUE     = safe_float(P_VALUE),
        Stars       = sig_stars(P_VALUE)
      )
  }

  # --- Essay 3: Informed Trading ---
  it <- store$e3_informed_trading
  if (!is.null(it) && nrow(it) > 0) {
    stats$essay3_informed_trading <- it |>
      mutate(
        METRIC_VALUE = safe_float(METRIC_VALUE),
        METRIC_PVAL  = safe_float(METRIC_PVAL),
        Stars        = sig_stars(METRIC_PVAL)
      )
  }

  # --- Essay 3: TOST ---
  tost <- store$e3_tost
  if (!is.null(tost) && nrow(tost) > 0) {
    stats$essay3_tost <- tost
  }

  # --- Essay 3: Placebo ---
  placebo <- store$e3_placebo
  if (!is.null(placebo) && nrow(placebo) > 0) {
    stats$essay3_placebo <- placebo
  }

  stats
}


# =============================================================================
# Significance Stars
# =============================================================================

#' Import sig_stars from visual.R or define locally
if (!exists("sig_stars")) {
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
}

if (!exists("safe_float")) {
  safe_float <- function(x, default = NA_real_) {
    result <- suppressWarnings(as.numeric(x))
    ifelse(is.na(result), default, result)
  }
}


#' Add significance stars column to a data frame
#'
#' @param df A data frame
#' @param p_col Character name of the p-value column
#' @param new_col Character name for the new stars column (default: "Stars")
#' @return The data frame with an added significance stars column
add_significance_stars <- function(df, p_col, new_col = "Stars") {
  if (!p_col %in% names(df)) {
    warning(glue("Column '{p_col}' not found in data frame"))
    return(df)
  }
  df[[new_col]] <- sig_stars(safe_float(df[[p_col]]))
  df
}


# =============================================================================
# Excel Workbook Generation
# =============================================================================

#' Generate a formatted Excel workbook with all dissertation results
#'
#' @param store Named list from load_result_store()
#' @param output_path Character path for the output .xlsx file
#' @param stats Optional pre-computed summary statistics list
#' @return Invisible path to the saved workbook
generate_workbook <- function(store, output_path = "output/dissertation_results.xlsx",
                              stats = NULL) {
  if (is.null(stats)) {
    stats <- summary_statistics(store)
  }

  wb <- createWorkbook()

  # -- Styling --
  header_style <- createStyle(
    fontName    = "Calibri",
    fontSize    = 11,
    fontColour  = "#FFFFFF",
    fgFill      = "#1B2A4A",
    halign      = "center",
    textDecoration = "bold",
    border      = "TopBottomLeftRight",
    borderColour = "#FFFFFF"
  )

  body_style <- createStyle(
    fontName = "Calibri",
    fontSize = 10,
    border   = "TopBottomLeftRight",
    borderColour = "#E0E0E0"
  )

  sig_style <- createStyle(
    fontName   = "Calibri",
    fontSize   = 10,
    fontColour = "#e74c3c",
    textDecoration = "bold"
  )

  number_style <- createStyle(
    fontName = "Calibri",
    fontSize = 10,
    numFmt   = "0.0000",
    border   = "TopBottomLeftRight",
    borderColour = "#E0E0E0"
  )

  pct_style <- createStyle(
    fontName = "Calibri",
    fontSize = 10,
    numFmt   = "0.00%",
    border   = "TopBottomLeftRight",
    borderColour = "#E0E0E0"
  )

  # Helper to add a sheet
  add_sheet <- function(wb, name, df) {
    if (is.null(df) || nrow(df) == 0) return(invisible(NULL))

    # Truncate sheet name to 31 chars (Excel limit)
    sheet_name <- substr(name, 1, 31)
    addWorksheet(wb, sheet_name)
    writeData(wb, sheet_name, df, headerStyle = header_style, startRow = 1)

    # Apply body style
    nr <- nrow(df)
    nc <- ncol(df)
    if (nr > 0 && nc > 0) {
      addStyle(wb, sheet_name, body_style,
               rows = 2:(nr + 1), cols = 1:nc,
               gridExpand = TRUE, stack = TRUE)

      # Format numeric columns
      num_cols <- which(sapply(df, is.numeric))
      if (length(num_cols) > 0) {
        addStyle(wb, sheet_name, number_style,
                 rows = 2:(nr + 1), cols = num_cols,
                 gridExpand = TRUE, stack = TRUE)
      }

      # Bold significance stars
      if ("Stars" %in% names(df)) {
        star_col <- which(names(df) == "Stars")
        star_rows <- which(df$Stars != "" & !is.na(df$Stars)) + 1
        if (length(star_rows) > 0) {
          addStyle(wb, sheet_name, sig_style,
                   rows = star_rows, cols = star_col,
                   stack = TRUE)
        }
      }
    }

    # Auto-width columns
    setColWidths(wb, sheet_name, cols = 1:nc, widths = "auto")

    # Freeze header row
    freezePane(wb, sheet_name, firstRow = TRUE)

    invisible(NULL)
  }

  # -- Add summary statistics sheets --
  message("Generating workbook...")
  message("  Adding summary statistics sheets...")

  for (sheet_name in names(stats)) {
    df <- stats[[sheet_name]]
    if (is.data.frame(df) && nrow(df) > 0) {
      add_sheet(wb, paste0("Summary_", sheet_name), df)
    }
  }

  # -- Add raw result tables --
  message("  Adding raw result tables...")

  raw_tables <- list(
    # Essay 1
    "E1_FF5_Coefficients"   = store$e1_ff5_coefficients,
    "E1_Factor_Premia"      = store$e1_factor_premia,
    "E1_Chow_Test"          = store$e1_chow_test,
    "E1_CW_Stock_Results"   = store$e1_cw_stock,
    "E1_FOMO_By_Regime"     = store$e1_fomo,
    "E1_Matched_TTest"      = store$e1_matched_ttest,
    "E1_Matched_Sign"       = store$e1_matched_sign,
    "E1_Matched_Amplify"    = store$e1_matched_amp,

    # Essay 2
    "E2_CAR_Panel"          = store$e2_car_panel,
    "E2_DiD_Coefficients"   = store$e2_did_coeff,
    "E2_Parallel_Trends"    = store$e2_parallel,
    "E2_MultiWindow_Summary"= store$e2_mw_summary,
    "E2_Contagion_Summary"  = store$e2_cont_summary,
    "E2_Alignment"          = store$e2_alignment,

    # Essay 3
    "E3_Informed_Trading"   = store$e3_informed_trading,
    "E3_Proximity"          = store$e3_informed_proximity,
    "E3_Size_Accuracy"      = store$e3_size_accuracy,
    "E3_TOST"               = store$e3_tost,
    "E3_Placebo"            = store$e3_placebo,
    "E3_Bootstrap_CI"       = store$e3_bootstrap_ci,
    "E3_Wilcoxon_Family"    = store$e3_wilcoxon_family,
    "E3_Repeat_Traders"     = store$e3_repeat_traders
  )

  for (sheet_name in names(raw_tables)) {
    df <- raw_tables[[sheet_name]]
    if (!is.null(df) && is.data.frame(df) && nrow(df) > 0) {
      # Add significance stars if P_VALUE column exists
      p_cols <- intersect(c("P_VALUE", "ALPHA_P", "METRIC_PVAL", "P_TOST"), names(df))
      if (length(p_cols) > 0) {
        df <- add_significance_stars(df, p_cols[1])
      }
      add_sheet(wb, sheet_name, df)
    }
  }

  # Save workbook
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  saveWorkbook(wb, output_path, overwrite = TRUE)
  message(glue("Workbook saved: {output_path}"))
  invisible(output_path)
}


# =============================================================================
# LaTeX Table Generation (bonus helper)
# =============================================================================

#' Generate a LaTeX-formatted coefficient table
#'
#' @param df A data frame with VARIABLE, COEFFICIENT, STD_ERROR, P_VALUE columns
#' @param caption LaTeX table caption
#' @param label LaTeX table label
#' @return Character string of LaTeX code
latex_coefficient_table <- function(df, caption = "Regression Coefficients",
                                   label = "tab:coefficients") {
  if (!all(c("VARIABLE", "COEFFICIENT") %in% names(df))) {
    warning("Expected columns: VARIABLE, COEFFICIENT, STD_ERROR, P_VALUE")
    return("")
  }

  df <- df |> mutate(
    COEFFICIENT = safe_float(COEFFICIENT),
    STD_ERROR   = safe_float(STD_ERROR %||% NA),
    P_VALUE     = safe_float(P_VALUE %||% NA),
    Stars       = sig_stars(P_VALUE)
  )

  header <- paste0(
    "\\begin{table}[htbp]\n",
    "\\centering\n",
    sprintf("\\caption{%s}\n", caption),
    sprintf("\\label{%s}\n", label),
    "\\begin{tabular}{lrrr}\n",
    "\\hline\n",
    "Variable & Coefficient & Std. Error & \\\\\n",
    "\\hline\n"
  )

  rows <- df |>
    rowwise() |>
    mutate(line = sprintf(
      "%s & %.4f%s & (%.4f) & \\\\",
      VARIABLE,
      COEFFICIENT,
      Stars,
      STD_ERROR
    )) |>
    pull(line) |>
    paste(collapse = "\n")

  footer <- paste0(
    "\n\\hline\n",
    "\\multicolumn{4}{l}{\\textit{Note}: $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, $^{\\dagger}p<0.10$} \\\\\n",
    "\\end{tabular}\n",
    "\\end{table}\n"
  )

  paste0(header, rows, footer)
}
