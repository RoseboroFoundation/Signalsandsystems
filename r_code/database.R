# =============================================================================
# Database Module for Signals & Systems Dissertation (R Version)
#
# Supports two backends:
#   - SQLite (primary) via DBI + RSQLite
#   - Athena (optional) via noctua
#
# Table naming convention: UPPERCASE_WITH_UNDERSCORES
#
# Usage:
#   source("database.R")
#   conn <- connect_db()
#   df   <- read_table(conn, "INFLATION_DATA")
#   write_table(conn, "MY_TABLE", my_tibble)
#   tables <- list_tables(conn)
#   result <- run_query(conn, "SELECT * FROM VIX_DATA LIMIT 10")
#   close_db(conn)
# =============================================================================

library(DBI)
library(RSQLite)
library(tibble)
library(dplyr)
library(readr)
library(purrr)
library(stringr)
library(lubridate)
library(glue)

# Optional Athena support — loaded on demand
# library(noctua)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

.DB_DEFAULT_PATH <- file.path(
  tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "."),
  "..", "data", "signals_systems.db"
)

# Normalise to an absolute path if possible
if (file.exists(dirname(.DB_DEFAULT_PATH))) {

  .DB_DEFAULT_PATH <- normalizePath(.DB_DEFAULT_PATH, mustWork = FALSE)
}

# Table mapping — mirrors Python TABLE_MAP keys to uppercase table names
TABLE_MAP <- list(

  # Market Data
  culturewardata              = list(table = "CULTURE_WAR_COMPANIES",   extract_mode = "dataframe"),
  stockdata                   = list(table = "STOCK_DATA",              extract_mode = "concat"),
  vixdata                     = list(table = "VIX_DATA",                extract_mode = "dataframe"),
  ff_factors                  = list(table = "FAMA_FRENCH_FACTORS",     extract_mode = "multi",
                                     sub_tables = list(FF3 = "FF3_FACTORS",
                                                       FF5 = "FF5_FACTORS",
                                                       MOM = "MOMENTUM_FACTORS")),
  newsdata                    = list(table = "NEWS_DATA",               extract_mode = "dataframe"),


  # Inflation
  inflationdata               = list(table = "INFLATION_DATA",          extract_mode = "combined"),
  inflation_expectations      = list(table = "INFLATION_EXPECTATIONS",  extract_mode = "combined"),
  comprehensive_inflation     = list(table = "INFLATION_COMPREHENSIVE", extract_mode = "combined"),


  # Interest Rates
  treasury_yields             = list(table = "TREASURY_YIELDS",         extract_mode = "combined"),
  policy_rates                = list(table = "POLICY_RATES",            extract_mode = "combined"),
  credit_spreads              = list(table = "CREDIT_SPREADS",          extract_mode = "combined"),
  comprehensive_rates         = list(table = "RATES_COMPREHENSIVE",     extract_mode = "combined"),


  # Industrial Production
  industrial_production       = list(table = "INDUSTRIAL_PRODUCTION",   extract_mode = "combined"),
  ip_growth                   = list(table = "IP_GROWTH_RATES",         extract_mode = "combined"),
  comprehensive_ip            = list(table = "IP_COMPREHENSIVE",        extract_mode = "combined"),

  # Money Supply
  money_supply                = list(table = "MONEY_SUPPLY",            extract_mode = "combined"),
  money_velocity              = list(table = "MONEY_VELOCITY",          extract_mode = "combined"),
  fed_balance_sheet           = list(table = "FED_BALANCE_SHEET",       extract_mode = "combined"),
  comprehensive_m2            = list(table = "M2_COMPREHENSIVE",        extract_mode = "combined"),

  # GDP
  gdp_data                    = list(table = "GDP_DATA",                extract_mode = "combined"),
  gdp_components              = list(table = "GDP_COMPONENTS",          extract_mode = "combined"),
  gdp_industry                = list(table = "GDP_INDUSTRY",            extract_mode = "combined"),
  comprehensive_gdp           = list(table = "GDP_COMPREHENSIVE",       extract_mode = "combined"),

  # Employment
  employment_data             = list(table = "EMPLOYMENT_DATA",         extract_mode = "combined"),
  jobless_claims              = list(table = "JOBLESS_CLAIMS",          extract_mode = "combined"),
  wages_hours                 = list(table = "WAGES_HOURS",             extract_mode = "combined"),
  jolts_data                  = list(table = "JOLTS_DATA",              extract_mode = "combined"),
  comprehensive_employment    = list(table = "EMPLOYMENT_COMPREHENSIVE",extract_mode = "combined"),

  # Additional Macro
  additional_macro            = list(table = "ADDITIONAL_MACRO",        extract_mode = "dataframe"),

  # Political
  political_events            = list(table = "POLITICAL_EVENTS",        extract_mode = "dataframe"),
  political_exposure          = list(table = "POLITICAL_EXPOSURE",      extract_mode = "dataframe"),

  # SEC / Insider

  form4data                   = list(table = "FORM4_TRANSACTIONS",      extract_mode = "dataframe"),
  sec_fundamentals            = list(table = "SEC_FUNDAMENTALS",        extract_mode = "dataframe"),
  controlcompanies            = list(table = "CONTROL_COMPANIES",       extract_mode = "dataframe"),

  # Essay 1
  essay1_regime_summary       = list(table = "ESSAY1_REGIME_SUMMARY",       extract_mode = "dataframe"),
  essay1_model_selection      = list(table = "ESSAY1_MODEL_SELECTION",      extract_mode = "dataframe"),
  essay1_transition_matrix    = list(table = "ESSAY1_TRANSITION_MATRIX",    extract_mode = "dataframe"),
  essay1_ff5_coefficients     = list(table = "ESSAY1_FF5_COEFFICIENTS",     extract_mode = "dataframe"),
  essay1_factor_premia        = list(table = "ESSAY1_FACTOR_PREMIA",        extract_mode = "dataframe"),
  essay1_chow_test            = list(table = "ESSAY1_CHOW_TEST",            extract_mode = "dataframe"),
  essay1_cw_stock_results     = list(table = "ESSAY1_CW_STOCK_RESULTS",     extract_mode = "dataframe"),
  essay1_sentiment_daily      = list(table = "ESSAY1_SENTIMENT_DAILY",      extract_mode = "dataframe"),
  essay1_fomo_by_regime       = list(table = "ESSAY1_FOMO_BY_REGIME",       extract_mode = "dataframe"),
  essay1_matched_deltas       = list(table = "ESSAY1_MATCHED_DELTAS",       extract_mode = "dataframe"),
  essay1_matched_ttest        = list(table = "ESSAY1_MATCHED_TTEST",        extract_mode = "dataframe"),
  essay1_matched_sign         = list(table = "ESSAY1_MATCHED_SIGN",         extract_mode = "dataframe"),
  essay1_matched_amplification= list(table = "ESSAY1_MATCHED_AMPLIFICATION",extract_mode = "dataframe"),
  essay1_matched_coverage     = list(table = "ESSAY1_MATCHED_COVERAGE",     extract_mode = "dataframe"),

  # Essay 2
  essay2_car_panel            = list(table = "ESSAY2_CAR_PANEL",            extract_mode = "dataframe"),
  essay2_did_coefficients     = list(table = "ESSAY2_DID_COEFFICIENTS",     extract_mode = "dataframe"),
  essay2_parallel_trends      = list(table = "ESSAY2_PARALLEL_TRENDS",      extract_mode = "dataframe"),
  essay2_news_sentiment       = list(table = "ESSAY2_NEWS_SENTIMENT",       extract_mode = "dataframe"),
  essay2_filing_sentiment     = list(table = "ESSAY2_FILING_SENTIMENT",     extract_mode = "dataframe"),
  essay2_event_nlp            = list(table = "ESSAY2_EVENT_NLP",            extract_mode = "dataframe"),
  essay2_political_alignment  = list(table = "ESSAY2_POLITICAL_ALIGNMENT",  extract_mode = "dataframe"),
  essay2_distinctive_phrases  = list(table = "ESSAY2_DISTINCTIVE_PHRASES",  extract_mode = "dataframe"),
  essay2_alignment_validation = list(table = "ESSAY2_ALIGNMENT_VALIDATION", extract_mode = "dataframe"),
  essay2_multi_window_summary = list(table = "ESSAY2_MULTI_WINDOW_SUMMARY", extract_mode = "dataframe"),
  essay2_multi_window_by_lean = list(table = "ESSAY2_MULTI_WINDOW_BY_LEAN", extract_mode = "dataframe"),
  essay2_multi_window_treat_vs_ctrl = list(table = "ESSAY2_MULTI_WINDOW_TREAT_VS_CTRL", extract_mode = "dataframe"),
  essay2_contagion_summary    = list(table = "ESSAY2_CONTAGION_SUMMARY",    extract_mode = "dataframe"),
  essay2_contagion_by_lean    = list(table = "ESSAY2_CONTAGION_BY_LEAN",    extract_mode = "dataframe"),
  essay2_contagion_by_facing  = list(table = "ESSAY2_CONTAGION_BY_FACING",  extract_mode = "dataframe"),
  essay2_contagion_peer_vs_nonpeer = list(table = "ESSAY2_CONTAGION_PEER_VS_NONPEER", extract_mode = "dataframe"),
  essay2_contagion_cons_vs_b2b= list(table = "ESSAY2_CONTAGION_CONS_VS_B2B",extract_mode = "dataframe"),
  essay2_contagion_lean_pairwise = list(table = "ESSAY2_CONTAGION_LEAN_PAIRWISE", extract_mode = "dataframe"),
  essay2_contagion_lean_mech  = list(table = "ESSAY2_CONTAGION_LEAN_MECH",  extract_mode = "dataframe"),
  essay2_contagion_tight_diff = list(table = "ESSAY2_CONTAGION_TIGHT_DIFF", extract_mode = "dataframe"),
  essay2_peer_parallel_trends = list(table = "ESSAY2_PEER_PARALLEL_TRENDS", extract_mode = "dataframe"),

  # Essay 3
  essay3_panel                = list(table = "ESSAY3_PANEL",                extract_mode = "dataframe"),
  essay3_informed_trading     = list(table = "ESSAY3_INFORMED_TRADING",     extract_mode = "dataframe"),
  essay3_informed_proximity   = list(table = "ESSAY3_INFORMED_PROXIMITY",   extract_mode = "dataframe"),
  essay3_informed_dollars     = list(table = "ESSAY3_INFORMED_DOLLARS",     extract_mode = "dataframe"),
  essay3_size_accuracy        = list(table = "ESSAY3_SIZE_ACCURACY",        extract_mode = "dataframe"),
  essay3_size_accuracy_slopes = list(table = "ESSAY3_SIZE_ACCURACY_SLOPES", extract_mode = "dataframe"),
  essay3_reversal_regression  = list(table = "ESSAY3_REVERSAL_REGRESSION",  extract_mode = "dataframe"),
  essay3_control_trades       = list(table = "ESSAY3_CONTROL_TRADES",       extract_mode = "dataframe"),
  essay3_crsp_profits         = list(table = "ESSAY3_CRSP_PROFITS",         extract_mode = "dataframe"),
  essay3_crsp_summary         = list(table = "ESSAY3_CRSP_SUMMARY",         extract_mode = "dataframe"),
  essay3_insider_panel        = list(table = "ESSAY3_INSIDER_PANEL",        extract_mode = "dataframe"),
  essay3_wilcoxon_family      = list(table = "ESSAY3_WILCOXON_FAMILY",      extract_mode = "dataframe"),
  essay3_insider_concentration= list(table = "ESSAY3_INSIDER_CONCENTRATION",extract_mode = "dataframe"),
  essay3_concentration_cuts   = list(table = "ESSAY3_CONCENTRATION_CUTS",   extract_mode = "dataframe"),
  essay3_tost                 = list(table = "ESSAY3_TOST",                 extract_mode = "dataframe"),
  essay3_placebo              = list(table = "ESSAY3_PLACEBO",              extract_mode = "dataframe"),
  essay3_bootstrap_ci         = list(table = "ESSAY3_BOOTSTRAP_CI",         extract_mode = "dataframe"),
  essay3_stratification       = list(table = "ESSAY3_STRATIFICATION",       extract_mode = "dataframe"),
  essay3_mean_vs_dist         = list(table = "ESSAY3_MEAN_VS_DISTRIBUTIONAL", extract_mode = "dataframe"),
  essay3_quantile_reg         = list(table = "ESSAY3_QUANTILE_REGRESSION",  extract_mode = "dataframe"),
  essay3_trimmed              = list(table = "ESSAY3_TRIMMED_ROBUSTNESS",   extract_mode = "dataframe"),
  essay3_bootstrap_wilcoxon   = list(table = "ESSAY3_BOOTSTRAP_WILCOXON",   extract_mode = "dataframe"),
  essay3_active_subset        = list(table = "ESSAY3_ACTIVE_SUBSET",        extract_mode = "dataframe"),
  essay3_repeat_traders       = list(table = "ESSAY3_REPEAT_TRADERS",       extract_mode = "dataframe")
)


# =============================================================================
# Connection Management
# =============================================================================

#' Connect to the database
#'
#' @param db_path Character path to the SQLite file (default: data/signals_systems.db)
#' @param backend Character, either "sqlite" or "athena"
#' @return A DBI connection object
connect_db <- function(db_path = NULL, backend = "sqlite") {
  if (backend == "athena") {
    return(connect_athena())
  }
  if (is.null(db_path)) {
    db_path <- .DB_DEFAULT_PATH
  }
  # Ensure parent directory exists

  dir.create(dirname(db_path), showWarnings = FALSE, recursive = TRUE)

  conn <- dbConnect(RSQLite::SQLite(), dbname = db_path)
  # Enable WAL mode for better concurrent access

  dbExecute(conn, "PRAGMA journal_mode=WAL")
  message(glue("Connected to SQLite: {db_path}"))
  return(conn)
}

#' Connect to Athena (optional — requires noctua package)
#'
#' @return A DBI connection via noctua
connect_athena <- function() {
  if (!requireNamespace("noctua", quietly = TRUE)) {
    stop("Package 'noctua' required for Athena backend. Install with: install.packages('noctua')")
  }
  aws_region <- Sys.getenv("AWS_REGION", "us-east-1")
  glue_db    <- Sys.getenv("GLUE_DATABASE", "roseboro_research")
  workgroup  <- Sys.getenv("ATHENA_WORKGROUP", "roseboro")
  s3_staging <- Sys.getenv("ATHENA_RESULTS_BUCKET", "s3://roseboro-athena-results/")

  conn <- noctua::dbConnect(
    noctua::athena(),
    schema_name   = glue_db,
    work_group    = workgroup,
    s3_staging_dir = s3_staging,
    region_name   = aws_region
  )
  message(glue("Connected to Athena: {glue_db} (workgroup={workgroup})"))
  return(conn)
}


# =============================================================================
# Read / Write / Query
# =============================================================================

#' Read a table into a tibble
#'
#' @param conn DBI connection
#' @param table_name Character table name (case-insensitive; uppercased internally)
#' @param limit Optional integer row limit
#' @return A tibble
read_table <- function(conn, table_name, limit = NULL) {
  tbl_name <- toupper(table_name)

  # Check table exists
  if (!dbExistsTable(conn, tbl_name)) {
    warning(glue("Table '{tbl_name}' does not exist"))
    return(tibble())
  }

  sql <- glue('SELECT * FROM "{tbl_name}"')
  if (!is.null(limit)) {
    sql <- glue("{sql} LIMIT {limit}")
  }

  df <- dbGetQuery(conn, sql)
  as_tibble(df)
}


#' Write a tibble to a table
#'
#' @param conn DBI connection
#' @param table_name Character table name (uppercased internally)
#' @param df A data frame or tibble to write
#' @param overwrite Logical; if TRUE, replaces the table. If FALSE, appends.
#' @return Invisible list with status info
write_table <- function(conn, table_name, df, overwrite = TRUE) {
  tbl_name <- toupper(table_name)
  t0 <- proc.time()

  # Prepare: uppercase column names, flatten dates
  write_df <- prepare_dataframe(df)

  tryCatch({
    dbWriteTable(
      conn, tbl_name, write_df,
      overwrite = overwrite,
      append    = !overwrite,
      row.names = FALSE
    )
    elapsed <- (proc.time() - t0)["elapsed"]
    message(glue("  [OK]   {format(tbl_name, width=30)} {nrow(write_df)} rows ({round(elapsed, 1)}s)"))
    invisible(list(
      table    = tbl_name,
      rows     = nrow(write_df),
      status   = "SUCCESS",
      duration = as.numeric(elapsed)
    ))
  }, error = function(e) {
    elapsed <- (proc.time() - t0)["elapsed"]
    warning(glue("  [FAIL] {tbl_name}: {conditionMessage(e)}"))
    invisible(list(
      table    = tbl_name,
      rows     = 0L,
      status   = "FAILED",
      error    = conditionMessage(e),
      duration = as.numeric(elapsed)
    ))
  })
}


#' List all tables in the database
#'
#' @param conn DBI connection
#' @return A tibble with columns: name, rows
list_tables <- function(conn) {
  tables <- dbListTables(conn)
  # Exclude internal sqlite tables
  tables <- tables[!grepl("^sqlite_", tables)]

  tibble(
    name = tables,
    rows = map_int(tables, ~ {
      tryCatch(
        dbGetQuery(conn, glue('SELECT COUNT(*) AS n FROM "{.x}"'))$n,
        error = function(e) NA_integer_
      )
    })
  ) |>
    arrange(name)
}


#' Run an arbitrary SQL query
#'
#' @param conn DBI connection
#' @param sql Character SQL string
#' @return A tibble of results
run_query <- function(conn, sql) {
  as_tibble(dbGetQuery(conn, sql))
}


#' Close the database connection
#'
#' @param conn DBI connection
close_db <- function(conn) {
  if (!is.null(conn) && dbIsValid(conn)) {
    dbDisconnect(conn)
    message("Database connection closed")
  }
}


# =============================================================================
# DataFrame Preparation (mirrors Python _prepare_dataframe)
# =============================================================================

#' Prepare a data frame for database ingestion
#'
#' - Uppercase column names
#' - Replace spaces/hyphens with underscores
#' - Convert Date/POSIXct columns to character ISO format
#' - Deduplicate column names
#'
#' @param df A data frame
#' @return A cleaned data frame
prepare_dataframe <- function(df) {
  write_df <- as.data.frame(df)

  # If the data frame has a Date/POSIXct rownames-style index, move it to a column
  if (inherits(rownames(write_df), "Date") || inherits(rownames(write_df), "POSIXct")) {
    write_df$DATE <- rownames(write_df)
    rownames(write_df) <- NULL
  }

  # Uppercase column names, replace spaces/hyphens
  names(write_df) <- names(write_df) |>
    toupper() |>
    str_replace_all(" ", "_") |>
    str_replace_all("-", "_")

  # Deduplicate column names
  seen <- list()
  new_names <- character(ncol(write_df))
  for (i in seq_along(names(write_df))) {
    nm <- names(write_df)[i]
    if (nm %in% names(seen)) {
      seen[[nm]] <- seen[[nm]] + 1L
      new_names[i] <- paste0(nm, "_", seen[[nm]])
    } else {
      seen[[nm]] <- 1L
      new_names[i] <- nm
    }
  }
  names(write_df) <- new_names

  # Convert Date and POSIXct columns to ISO character strings
  for (col in names(write_df)) {
    if (inherits(write_df[[col]], "POSIXct")) {
      write_df[[col]] <- format(write_df[[col]], "%Y-%m-%d %H:%M:%S")
    } else if (inherits(write_df[[col]], "Date")) {
      write_df[[col]] <- as.character(write_df[[col]])
    }
  }

  write_df
}


# =============================================================================
# Data Extraction from ETL list (mirrors Python _extract_dataframe)
# =============================================================================

#' Extract a data frame from an ETL data entry
#'
#' @param data The ETL value (data.frame or list)
#' @param extract_mode One of "dataframe", "combined", "concat"
#' @return A data frame or NULL
extract_dataframe <- function(data, extract_mode) {
  if (extract_mode == "dataframe") {
    if (is.data.frame(data)) return(data)
    return(NULL)
  }

  if (extract_mode == "combined") {
    if (is.list(data) && !is.data.frame(data)) {
      if ("combined" %in% names(data) && is.data.frame(data$combined)) {
        return(data$combined)
      }
      # Return first data frame found
      for (v in data) {
        if (is.data.frame(v)) return(v)
      }
    }
    if (is.data.frame(data)) return(data)
    return(NULL)
  }

  if (extract_mode == "concat") {
    if (is.list(data) && !is.data.frame(data)) {
      frames <- keep(data, is.data.frame)
      frames <- keep(frames, ~ nrow(.x) > 0)
      if (length(frames) > 0) {
        return(bind_rows(frames))
      }
    }
    return(NULL)
  }

  NULL
}


# =============================================================================
# Bulk Load (mirrors Python load_to_sqlite)
# =============================================================================

#' Bulk load all ETL data into SQLite
#'
#' @param data_list Named list of data frames / nested lists (output from ETL)
#' @param db_path Character path to the SQLite database
#' @param replace Logical; replace existing tables?
#' @return A tibble summarising the load results
load_to_sqlite <- function(data_list, db_path = NULL, replace = TRUE) {
  conn <- connect_db(db_path = db_path, backend = "sqlite")
  on.exit(close_db(conn), add = TRUE)

  results <- list()
  t_start <- proc.time()

  message(strrep("=", 60))
  message("  Loading ETL data into SQLite")
  message(glue("  Database: {db_path %||% .DB_DEFAULT_PATH}"))
  message(glue("  Mode: {ifelse(replace, 'REPLACE', 'APPEND')}"))
  message(strrep("=", 60))

  for (etl_key in names(data_list)) {
    data <- data_list[[etl_key]]
    if (is.null(data)) {
      message(glue("  [SKIP] {format(etl_key, width=30)} (no data)"))
      next
    }

    mapping <- TABLE_MAP[[etl_key]]
    if (is.null(mapping)) {
      message(glue("  [SKIP] {format(etl_key, width=30)} (no table mapping)"))
      next
    }

    extract_mode <- mapping$extract_mode

    if (extract_mode == "multi") {
      sub_tables <- mapping$sub_tables
      for (sub_key in names(sub_tables)) {
        sub_table <- sub_tables[[sub_key]]
        sub_df <- if (is.list(data) && !is.data.frame(data)) data[[sub_key]] else NULL
        if (!is.null(sub_df) && is.data.frame(sub_df) && nrow(sub_df) > 0) {
          res <- write_table(conn, sub_table, sub_df, overwrite = replace)
          results[[sub_table]] <- res
        } else {
          message(glue("  [SKIP] {format(sub_table, width=30)} (sub-key '{sub_key}' empty)"))
        }
      }
    } else {
      df <- extract_dataframe(data, extract_mode)
      if (!is.null(df) && nrow(df) > 0) {
        table_name <- mapping$table
        res <- write_table(conn, table_name, df, overwrite = replace)
        results[[table_name]] <- res
      } else {
        message(glue("  [SKIP] {format(etl_key, width=30)} (extracted empty)"))
      }
    }
  }

  # Log load metadata
  log_load_metadata(conn, results)

  elapsed <- (proc.time() - t_start)["elapsed"]
  succeeded <- keep(results, ~ .x$status == "SUCCESS")
  failed    <- keep(results, ~ .x$status != "SUCCESS")
  total_rows <- sum(map_int(succeeded, "rows"))

  message("")
  message(strrep("=", 60))
  message("  SQLite Load Summary")
  message(strrep("=", 60))
  message(glue("  Tables loaded:  {length(succeeded)}"))
  message(glue("  Tables failed:  {length(failed)}"))
  message(glue("  Total rows:     {format(total_rows, big.mark = ',')}"))
  message(glue("  Total time:     {round(elapsed, 1)}s"))
  if (file.exists(db_path %||% .DB_DEFAULT_PATH)) {
    db_mb <- file.info(db_path %||% .DB_DEFAULT_PATH)$size / 1024 / 1024
    message(glue("  Database size:  {round(db_mb, 1)} MB"))
  }
  message(strrep("=", 60))

  bind_rows(results) |> as_tibble()
}


#' Log load metadata to ETL_LOAD_LOG table
#'
#' @param conn DBI connection
#' @param results List of load result lists
log_load_metadata <- function(conn, results) {
  tryCatch({
    dbExecute(conn, "
      CREATE TABLE IF NOT EXISTS ETL_LOAD_LOG (
        LOAD_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        LOAD_TIMESTAMP TEXT DEFAULT (datetime('now')),
        TABLE_NAME TEXT,
        ROWS_LOADED INTEGER,
        STATUS TEXT,
        ERROR_MESSAGE TEXT,
        DURATION_SECONDS REAL
      )
    ")
    for (res in results) {
      dbExecute(conn, "
        INSERT INTO ETL_LOAD_LOG
          (LOAD_TIMESTAMP, TABLE_NAME, ROWS_LOADED, STATUS, ERROR_MESSAGE, DURATION_SECONDS)
        VALUES (datetime('now'), ?, ?, ?, ?, ?)
      ", params = list(
        res$table, res$rows, res$status,
        res$error %||% NA_character_,
        round(res$duration, 2)
      ))
    }
  }, error = function(e) {
    warning(glue("Failed to log load metadata: {conditionMessage(e)}"))
  })
}


# =============================================================================
# ResultStore — load all essay results (mirrors Python ResultStore)
# =============================================================================

#' Load all essay results from the database into a named list
#'
#' @param conn DBI connection (or NULL to auto-connect)
#' @param db_path Optional SQLite path (used if conn is NULL)
#' @return A named list of tibbles
load_result_store <- function(conn = NULL, db_path = NULL) {
  auto_close <- is.null(conn)
  if (is.null(conn)) {
    conn <- connect_db(db_path = db_path)
  }

  safe_read <- function(tbl) {
    tryCatch(
      read_table(conn, tbl),
      error = function(e) tibble()
    )
  }

  store <- list(
    # Essay 1
    e1_ff5_coefficients   = safe_read("ESSAY1_FF5_COEFFICIENTS"),
    e1_factor_premia      = safe_read("ESSAY1_FACTOR_PREMIA"),
    e1_chow_test          = safe_read("ESSAY1_CHOW_TEST"),
    e1_cw_stock           = safe_read("ESSAY1_CW_STOCK_RESULTS"),
    e1_sentiment          = safe_read("ESSAY1_SENTIMENT_DAILY"),
    e1_fomo               = safe_read("ESSAY1_FOMO_BY_REGIME"),
    e1_matched_deltas     = safe_read("ESSAY1_MATCHED_DELTAS"),
    e1_matched_ttest      = safe_read("ESSAY1_MATCHED_TTEST"),
    e1_matched_sign       = safe_read("ESSAY1_MATCHED_SIGN"),
    e1_matched_amp        = safe_read("ESSAY1_MATCHED_AMPLIFICATION"),
    e1_matched_coverage   = safe_read("ESSAY1_MATCHED_COVERAGE"),
    vix_data              = safe_read("VIX_DATA"),
    cw_companies          = safe_read("CULTURE_WAR_COMPANIES"),

    # Essay 2
    e2_car_panel          = safe_read("ESSAY2_CAR_PANEL"),
    e2_did_coeff          = safe_read("ESSAY2_DID_COEFFICIENTS"),
    e2_parallel           = safe_read("ESSAY2_PARALLEL_TRENDS"),
    e2_news_sent          = safe_read("ESSAY2_NEWS_SENTIMENT"),
    e2_filing_sent        = safe_read("ESSAY2_FILING_SENTIMENT"),
    e2_event_nlp          = safe_read("ESSAY2_EVENT_NLP"),
    e2_alignment          = safe_read("ESSAY2_POLITICAL_ALIGNMENT"),
    e2_phrases            = safe_read("ESSAY2_DISTINCTIVE_PHRASES"),
    e2_validation         = safe_read("ESSAY2_ALIGNMENT_VALIDATION"),
    e2_mw_summary         = safe_read("ESSAY2_MULTI_WINDOW_SUMMARY"),
    e2_mw_lean            = safe_read("ESSAY2_MULTI_WINDOW_BY_LEAN"),
    e2_mw_tc              = safe_read("ESSAY2_MULTI_WINDOW_TREAT_VS_CTRL"),
    e2_cont_summary       = safe_read("ESSAY2_CONTAGION_SUMMARY"),
    e2_cont_lean          = safe_read("ESSAY2_CONTAGION_BY_LEAN"),
    e2_cont_facing        = safe_read("ESSAY2_CONTAGION_BY_FACING"),
    e2_cont_peer          = safe_read("ESSAY2_CONTAGION_PEER_VS_NONPEER"),
    e2_cont_cb            = safe_read("ESSAY2_CONTAGION_CONS_VS_B2B"),
    e2_cont_lp            = safe_read("ESSAY2_CONTAGION_LEAN_PAIRWISE"),
    e2_cont_mech          = safe_read("ESSAY2_CONTAGION_LEAN_MECH"),
    e2_cont_tight         = safe_read("ESSAY2_CONTAGION_TIGHT_DIFF"),
    e2_peer_parallel      = safe_read("ESSAY2_PEER_PARALLEL_TRENDS"),

    # Essay 3
    e3_panel              = safe_read("ESSAY3_PANEL"),
    e3_informed_trading   = safe_read("ESSAY3_INFORMED_TRADING"),
    e3_informed_proximity = safe_read("ESSAY3_INFORMED_PROXIMITY"),
    e3_informed_dollars   = safe_read("ESSAY3_INFORMED_DOLLARS"),
    e3_size_accuracy      = safe_read("ESSAY3_SIZE_ACCURACY"),
    e3_size_accuracy_slopes = safe_read("ESSAY3_SIZE_ACCURACY_SLOPES"),
    e3_reversal_regression= safe_read("ESSAY3_REVERSAL_REGRESSION"),
    e3_control_trades     = safe_read("ESSAY3_CONTROL_TRADES"),
    e3_crsp_profits       = safe_read("ESSAY3_CRSP_PROFITS"),
    e3_crsp_summary       = safe_read("ESSAY3_CRSP_SUMMARY"),
    e3_insider_panel      = safe_read("ESSAY3_INSIDER_PANEL"),
    e3_wilcoxon_family    = safe_read("ESSAY3_WILCOXON_FAMILY"),
    e3_insider_concentration = safe_read("ESSAY3_INSIDER_CONCENTRATION"),
    e3_concentration_cuts = safe_read("ESSAY3_CONCENTRATION_CUTS"),
    e3_tost               = safe_read("ESSAY3_TOST"),
    e3_placebo            = safe_read("ESSAY3_PLACEBO"),
    e3_bootstrap_ci       = safe_read("ESSAY3_BOOTSTRAP_CI"),
    e3_stratification     = safe_read("ESSAY3_STRATIFICATION"),
    e3_mean_vs_dist       = safe_read("ESSAY3_MEAN_VS_DISTRIBUTIONAL"),
    e3_quantile_reg       = safe_read("ESSAY3_QUANTILE_REGRESSION"),
    e3_trimmed            = safe_read("ESSAY3_TRIMMED_ROBUSTNESS"),
    e3_bootstrap_wilcoxon = safe_read("ESSAY3_BOOTSTRAP_WILCOXON"),
    e3_active_subset      = safe_read("ESSAY3_ACTIVE_SUBSET"),
    e3_repeat_traders     = safe_read("ESSAY3_REPEAT_TRADERS")
  )

  if (auto_close) close_db(conn)
  message("ResultStore loaded")
  store
}
