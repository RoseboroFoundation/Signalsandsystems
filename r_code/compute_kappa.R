# =============================================================================
# compute_kappa.R — Cohen's Kappa Inter-Rater Reliability
# Signals & Systems Dissertation Pipeline
#
# Computes Cohen's kappa between coder1 and coder2 initiation codings.
# R equivalent of compute_kappa.py
# =============================================================================

library(tidyverse)
library(irr)

# =============================================================================
# COHEN'S KAPPA
# =============================================================================

#' Compute Cohen's Kappa for two categorical vectors
#'
#' @param y1 Vector of coder 1 ratings
#' @param y2 Vector of coder 2 ratings
#' @return Named list with kappa, observed agreement, and confusion matrix
cohens_kappa <- function(y1, y2) {
  if (length(y1) == 0 || length(y2) == 0) {
    return(list(kappa = NA, p_observed = NA, confusion_matrix = NULL))
  }

  # Get all unique labels

labels <- sort(unique(c(y1, y2)))

  # Confusion matrix
  cm <- table(
    factor(y1, levels = labels),
    factor(y2, levels = labels),
    dnn = c("coder1", "coder2")
  )

  n <- length(y1)

  # Observed agreement
  p_o <- sum(diag(cm)) / n

  # Expected agreement (by chance)
  row_marginals <- rowSums(cm) / n
  col_marginals <- colSums(cm) / n
  p_e <- sum(row_marginals * col_marginals)

  # Kappa
  if (p_e == 1.0) {
    kappa_val <- 1.0
  } else {
    kappa_val <- (p_o - p_e) / (1.0 - p_e)
  }

  list(
    kappa = kappa_val,
    p_observed = p_o,
    confusion_matrix = cm
  )
}


#' Compute Cohen's Kappa using the irr package (alternative)
#'
#' @param y1 Vector of coder 1 ratings
#' @param y2 Vector of coder 2 ratings
#' @return kappa2 result object
cohens_kappa_irr <- function(y1, y2) {
  ratings <- cbind(y1, y2)
  irr::kappa2(ratings)
}


# =============================================================================
# MAIN
# =============================================================================

run_kappa_analysis <- function(csv_path = "event_initiation_coding.csv") {
  df <- read_csv(csv_path, show_col_types = FALSE)

  # Filter to rows where both coders have filled in initiation
  coded <- df %>%
    filter(!is.na(coder2_initiation) & coder2_initiation != "")

  cat(sprintf("Rows with both codings: %d / %d\n", nrow(coded), nrow(df)))

  if (nrow(coded) == 0) {
    cat("No coder2 data yet. Fill coder2_initiation column and rerun.\n")
    return(invisible(NULL))
  }

  y1 <- coded$initiation
  y2 <- coded$coder2_initiation

  result <- cohens_kappa(y1, y2)
  cat(sprintf("\nCohen's kappa: %.4f\n", result$kappa))
  cat(sprintf("Observed agreement: %.4f\n", result$p_observed))
  cat("\nConfusion matrix:\n")
  print(result$confusion_matrix)

  # Also compute kappa for lead_time if both filled
  lt_coded <- coded %>%
    filter(!is.na(coder2_lead_time) & coder2_lead_time != "")

  if (nrow(lt_coded) > 0) {
    lt_result <- cohens_kappa(lt_coded$lead_time, lt_coded$coder2_lead_time)
    cat(sprintf("\nLead-time kappa: %.4f (N=%d)\n", lt_result$kappa, nrow(lt_coded)))
    print(lt_result$confusion_matrix)
  }

  invisible(result)
}


# Run if executed directly
if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  csv_path <- if (length(args) > 0) args[1] else "event_initiation_coding.csv"
  run_kappa_analysis(csv_path)
}
