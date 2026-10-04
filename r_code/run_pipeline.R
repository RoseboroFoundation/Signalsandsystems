# =============================================================================
# Master Pipeline Script for Signals & Systems Dissertation (R Version)
#
# Sources all R modules and runs the full pipeline:
#   1. Connect to database
#   2. Load result store (all essay results)
#   3. Generate summary statistics
#   4. Generate all visualizations
#   5. Generate Excel workbook
#
# Prerequisites:
#   - Python ETL pipeline must have been run first (populates SQLite DB)
#   - Required R packages: DBI, RSQLite, dplyr, tidyr, tibble, purrr,
#     stringr, lubridate, glue, ggplot2, scales, openxlsx, readr
#
# Optional packages: patchwork (multi-panel plots), noctua (Athena backend)
#
# Usage:
#   Rscript run_pipeline.R                    # run full pipeline
#   Rscript run_pipeline.R --essay 1          # run Essay 1 only
#   Rscript run_pipeline.R --no-save          # skip file saving
#   Rscript run_pipeline.R --db path/to.db    # custom database path
#
# Or in an interactive R session:
#   source("run_pipeline.R")
#   results <- main()
# =============================================================================

# =============================================================================
# Source all modules
# =============================================================================

# Determine script directory for sourcing
.script_dir <- if (exists("ofile", envir = sys.frame(1))) {
  dirname(sys.frame(1)$ofile)
} else {
  "."
}

source(file.path(.script_dir, "database.R"))
source(file.path(.script_dir, "visual.R"))
source(file.path(.script_dir, "reporting.R"))

# Source additional R essay modules if they exist
.optional_sources <- c(
  "clean.R", "etl.R",
  "essay1.R", "essay1_matched.R",
  "essay2.R", "essay2_did.R",
  "essay3.R",
  "compute_kappa.R"
)

for (.src in .optional_sources) {
  .path <- file.path(.script_dir, .src)
  if (file.exists(.path)) {
    tryCatch(
      source(.path),
      error = function(e) message(glue("  [SKIP] Could not source {.src}: {conditionMessage(e)}"))
    )
  }
}

rm(.optional_sources, .src, .path)


# =============================================================================
# Parse command-line arguments
# =============================================================================

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- list(
    essays   = NULL,      # NULL = all essays
    save     = TRUE,
    db_path  = NULL,
    workbook = "output/dissertation_results.xlsx"
  )

  i <- 1
  while (i <= length(args)) {
    arg <- args[i]
    if (arg == "--essay" && i < length(args)) {
      i <- i + 1
      opts$essays <- as.integer(args[i])
    } else if (arg == "--no-save") {
      opts$save <- FALSE
    } else if (arg == "--db" && i < length(args)) {
      i <- i + 1
      opts$db_path <- args[i]
    } else if (arg == "--workbook" && i < length(args)) {
      i <- i + 1
      opts$workbook <- args[i]
    }
    i <- i + 1
  }
  opts
}


# =============================================================================
# Main Pipeline Function
# =============================================================================

#' Run the full Signals & Systems R pipeline
#'
#' @param db_path Optional SQLite database path
#' @param essays Integer vector of essay numbers to process (NULL = all)
#' @param save Logical; save figures and workbook?
#' @param workbook_path Path for the output Excel workbook
#' @return A list with results from each step
main <- function(db_path = NULL, essays = NULL, save = TRUE,
                 workbook_path = "output/dissertation_results.xlsx") {

  message(strrep("=", 60))
  message("  Signals & Systems -- R Pipeline")
  message(glue("  {format(Sys.time(), '%Y-%m-%d %H:%M:%S %Z')}"))
  message(strrep("=", 60))

  t_start <- proc.time()
  results <- list()

  # ── Step 1: Connect to Database ──────────────────────────────────────────
  message("\n[Step 1/5] Connecting to database...")
  conn <- tryCatch(
    connect_db(db_path = db_path),
    error = function(e) {
      message(glue("  ERROR: Could not connect: {conditionMessage(e)}"))
      return(NULL)
    }
  )

  if (is.null(conn)) {
    message("Pipeline aborted: no database connection.")
    return(invisible(results))
  }
  on.exit(close_db(conn), add = TRUE)

  # ── Step 2: Load Result Store ────────────────────────────────────────────
  message("\n[Step 2/5] Loading result store...")
  store <- tryCatch(
    load_result_store(conn),
    error = function(e) {
      message(glue("  ERROR: {conditionMessage(e)}"))
      return(NULL)
    }
  )

  if (is.null(store)) {
    message("Pipeline aborted: could not load result store.")
    return(invisible(results))
  }

  # Count loaded tables
  n_loaded <- sum(map_int(store, ~ if (is.data.frame(.x)) nrow(.x) else 0L) > 0)
  message(glue("  Loaded {n_loaded} / {length(store)} result tables"))
  results$n_tables_loaded <- n_loaded

  # ── Step 3: Summary Statistics ───────────────────────────────────────────
  message("\n[Step 3/5] Computing summary statistics...")
  stats <- tryCatch(
    summary_statistics(store),
    error = function(e) {
      message(glue("  ERROR: {conditionMessage(e)}"))
      list()
    }
  )
  results$summary_stats <- stats
  message(glue("  Generated {length(stats)} summary tables"))

  # Print key stats to console
  if ("events" %in% names(stats)) {
    evt <- stats$events
    message(glue("  Events: {evt$Total_Events} total, {evt$Unique_Tickers} tickers"))
  }
  if ("vix" %in% names(stats)) {
    vix <- stats$vix
    message(glue("  VIX: {vix$Observations} obs, mean={vix$Mean}, range=[{vix$Min}, {vix$Max}]"))
  }

  # ── Step 4: Generate Visualizations ──────────────────────────────────────
  message("\n[Step 4/5] Generating visualizations...")

  chart_list <- ALL_CHARTS
  if (!is.null(essays)) {
    if (1 %in% essays) {
      message("  -- Essay 1 charts --")
      chart_list <- ESSAY1_CHARTS
    } else if (2 %in% essays) {
      message("  -- Essay 2 charts --")
      chart_list <- ESSAY2_CHARTS
    } else if (3 %in% essays) {
      message("  -- Essay 3 charts --")
      chart_list <- ESSAY3_CHARTS
    }
  }

  chart_results <- tryCatch(
    generate_all_charts(store, chart_list = chart_list, save = save),
    error = function(e) {
      message(glue("  ERROR: {conditionMessage(e)}"))
      tibble()
    }
  )
  results$charts <- chart_results

  # ── Step 5: Generate Workbook ────────────────────────────────────────────
  if (save) {
    message("\n[Step 5/5] Generating Excel workbook...")
    wb_path <- tryCatch(
      generate_workbook(store, output_path = workbook_path, stats = stats),
      error = function(e) {
        message(glue("  ERROR: {conditionMessage(e)}"))
        NA_character_
      }
    )
    results$workbook_path <- wb_path
  } else {
    message("\n[Step 5/5] Skipping workbook (--no-save)")
  }

  # ── Summary ──────────────────────────────────────────────────────────────
  elapsed <- (proc.time() - t_start)["elapsed"]
  results$total_time <- as.numeric(elapsed)

  message("")
  message(strrep("=", 60))
  message("  Pipeline Complete")
  message(strrep("=", 60))
  message(glue("  Tables loaded:   {n_loaded}"))
  message(glue("  Summary tables:  {length(stats)}"))

  if (nrow(chart_results) > 0) {
    n_ok   <- sum(chart_results$status == "SUCCESS")
    n_fail <- sum(chart_results$status == "FAILED")
    message(glue("  Charts:          {n_ok} succeeded, {n_fail} failed"))
  }

  if (!is.null(results$workbook_path) && !is.na(results$workbook_path)) {
    message(glue("  Workbook:        {results$workbook_path}"))
  }

  if (save) {
    message(glue("  Figures dir:     {FIGURE_DIR}"))
  }

  message(glue("  Total time:      {round(elapsed, 1)}s"))
  message(strrep("=", 60))

  invisible(results)
}


# =============================================================================
# Run from command line
# =============================================================================

if (!interactive()) {
  opts <- parse_args()
  main(
    db_path       = opts$db_path,
    essays        = opts$essays,
    save          = opts$save,
    workbook_path = opts$workbook
  )
}
