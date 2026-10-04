# =============================================================================
# etl.R — ETL (Extract, Transform, Load) Pipeline for Signals & Systems
#
# R port of ETL.py. Loads all datasets defined in the data dictionary into
# a unified data store with selective loading, dependency resolution, and
# status reporting.
#
# Usage:
#   source("etl.R")
#
#   # Load everything
#   data <- run_etl()
#
#   # Load specific categories
#   data <- run_etl(categories = c("inflation", "rates", "employment"))
#
#   # Load specific keys
#   data <- run_etl(keys = c("inflationdata", "treasury_yields"))
#
#   # Quick summary
#   summarize_data(data)
# =============================================================================

# Source the clean module (must be in same directory or provide full path)
script_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) ".")
if (!file.exists(file.path(script_dir, "clean.R"))) {
  script_dir <- dirname(sys.argv <- commandArgs(trailingOnly = FALSE))
  idx <- grep("--file=", script_dir)
  if (length(idx) > 0) script_dir <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[idx[1]]))
  else script_dir <- "."
}
source(file.path(script_dir, "clean.R"))

# =============================================================================
# CONSTANTS
# =============================================================================

ETL_START_DATE <- "2000-01-01"
ETL_END_DATE   <- "2025-12-31"
ETL_CACHE_PATH <- "./data/fred"
CW_CSV_PATH    <- "Culture_War_Companies_160_fullmeta.csv"

# =============================================================================
# DATA DICTIONARY
# =============================================================================
# Each entry maps a dataset key to:
#   loader      — function to call
#   args        — named list of arguments
#   category    — grouping label
#   description — human-readable description
#   depends_on  — character vector of prerequisite keys

DATA_DICTIONARY <- list(

  # --- Culture War & Market Data ---
  culturewardata = list(
    loader      = "import_culture_war_data",
    args        = list(file_path = CW_CSV_PATH),
    category    = "market",
    description = "Culture war companies events dataset",
    depends_on  = character(0)
  ),

  stockdata = list(
    loader      = ".etl_load_stockdata",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE),
    category    = "market",
    description = "Historical stock prices from Yahoo Finance",
    depends_on  = "culturewardata"
  ),

  vixdata = list(
    loader      = "download_vix_data",
    args        = list(save_csv = TRUE),
    category    = "market",
    description = "CBOE Volatility Index (VIX) from FRED",
    depends_on  = character(0)
  ),

  ff_factors = list(
    loader      = "download_fama_french_factors",
    args        = list(
      start_date = ETL_START_DATE,
      frequency  = "daily",
      output_dir = "./fama_french_data"
    ),
    category    = "market",
    description = "Fama-French 3-factor, 5-factor, and Momentum",
    depends_on  = character(0)
  ),

  controlcompanies = list(
    loader      = ".etl_load_controlcompanies",
    args        = list(),
    category    = "market",
    description = "Control companies matched to culture war treatment firms",
    depends_on  = "culturewardata"
  ),

  industry_portfolios = list(
    loader      = "download_industry_portfolios",
    args        = list(
      num_industries = 10,
      start_date     = ETL_START_DATE,
      frequency      = "daily",
      output_dir     = "./fama_french_data"
    ),
    category    = "market",
    description = "Fama-French industry portfolio returns",
    depends_on  = character(0)
  ),

  # --- Political ---
  political_events = list(
    loader      = "load_political_events",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE),
    category    = "political",
    description = "Political events (congressional votes, EOs, court decisions)",
    depends_on  = character(0)
  ),

  political_exposure = list(
    loader      = ".etl_load_political_exposure",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE),
    category    = "political",
    description = "Firm-level political exposure (lobbying, PAC contributions)",
    depends_on  = "culturewardata"
  ),

  # --- Inflation ---
  inflationdata = list(
    loader      = "load_inflation_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "inflation",
    description = "Core inflation measures (CPI, PCE, PPI, GDP Deflator)",
    depends_on  = character(0)
  ),

  inflation_expectations = list(
    loader      = "load_inflation_expectations_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "inflation",
    description = "Breakeven inflation, survey expectations, Fed measures",
    depends_on  = character(0)
  ),

  comprehensive_inflation = list(
    loader      = "load_comprehensive_inflation_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "inflation",
    description = "All inflation measures combined with component-level CPI",
    depends_on  = character(0)
  ),

  # --- Interest Rates ---
  treasury_yields = list(
    loader      = "load_treasury_yields",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "rates",
    description = "Treasury yield curve (1M-30Y) and TIPS real yields",
    depends_on  = character(0)
  ),

  policy_rates = list(
    loader      = "load_policy_rates",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "rates",
    description = "Fed Funds, SOFR, Prime, Discount rates",
    depends_on  = character(0)
  ),

  credit_spreads = list(
    loader      = "load_credit_spreads",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "rates",
    description = "Corporate yields, credit spreads, mortgage rates",
    depends_on  = character(0)
  ),

  comprehensive_rates = list(
    loader      = "load_comprehensive_rates_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "rates",
    description = "All rates with yield curve metrics (slope, curvature, inversion)",
    depends_on  = character(0)
  ),

  # --- Industrial Production ---
  industrial_production = list(
    loader      = "load_industrial_production_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "production",
    description = "IP indices, sector production, capacity utilization",
    depends_on  = character(0)
  ),

  ip_growth = list(
    loader      = "load_ip_growth_rates",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "production",
    description = "IP growth rates (YoY, MoM) and diffusion indices",
    depends_on  = character(0)
  ),

  comprehensive_ip = list(
    loader      = "load_comprehensive_ip_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "production",
    description = "All industrial production measures combined",
    depends_on  = character(0)
  ),

  # --- Money Supply ---
  money_supply = list(
    loader      = "load_money_supply_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "money",
    description = "M1, M2, monetary base, and components",
    depends_on  = character(0)
  ),

  money_velocity = list(
    loader      = "load_money_velocity_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "money",
    description = "M1 and M2 velocity of money",
    depends_on  = character(0)
  ),

  fed_balance_sheet = list(
    loader      = "load_fed_balance_sheet_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "money",
    description = "Fed total assets, Treasury/MBS holdings, reserves",
    depends_on  = character(0)
  ),

  comprehensive_m2 = list(
    loader      = "load_comprehensive_m2_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "money",
    description = "All money supply measures with growth rates",
    depends_on  = character(0)
  ),

  # --- GDP ---
  gdp_data = list(
    loader      = "load_gdp_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "gdp",
    description = "Nominal/Real GDP, growth rates, per capita",
    depends_on  = character(0)
  ),

  gdp_components = list(
    loader      = "load_gdp_components_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "gdp",
    description = "GDP = C + I + G + (X-M) expenditure components",
    depends_on  = character(0)
  ),

  gdp_industry = list(
    loader      = "load_gdp_by_industry_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "gdp",
    description = "GDP by industry/sector (value added)",
    depends_on  = character(0)
  ),

  comprehensive_gdp = list(
    loader      = "load_comprehensive_gdp_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "gdp",
    description = "All GDP measures combined",
    depends_on  = character(0)
  ),

  # --- Employment ---
  employment_data = list(
    loader      = "load_employment_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "employment",
    description = "Payrolls, unemployment rates, labor force participation",
    depends_on  = character(0)
  ),

  jobless_claims = list(
    loader      = "load_jobless_claims_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "employment",
    description = "Initial/continuing claims, insured unemployment rate",
    depends_on  = character(0)
  ),

  wages_hours = list(
    loader      = "load_wages_hours_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "employment",
    description = "Average earnings, weekly hours, ECI, unit labor costs",
    depends_on  = character(0)
  ),

  jolts_data = list(
    loader      = "load_jolts_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "employment",
    description = "Job openings, hires, quits, layoffs (JOLTS)",
    depends_on  = character(0)
  ),

  comprehensive_employment = list(
    loader      = "load_comprehensive_employment_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "employment",
    description = "All employment measures combined",
    depends_on  = character(0)
  ),

  # --- Additional Macro ---
  additional_macro = list(
    loader      = "load_additional_macro_data",
    args        = list(start_date = ETL_START_DATE, end_date = ETL_END_DATE,
                       cache_path = ETL_CACHE_PATH),
    category    = "macro",
    description = "Consumer Sentiment, Housing Starts, Home Prices, Dollar Index",
    depends_on  = character(0)
  )
)

# Category descriptions for reporting
CATEGORIES <- list(
  market     = "Market Data (stocks, VIX, Fama-French, industry portfolios)",
  political  = "Political (events, lobbying exposure)",
  inflation  = "Inflation (CPI, PCE, PPI, expectations, components)",
  rates      = "Interest Rates (Treasury curve, policy rates, credit spreads)",
  production = "Industrial Production (IP indices, capacity utilization)",
  money      = "Money Supply (M1, M2, velocity, Fed balance sheet)",
  gdp        = "GDP (headline, components, by industry)",
  employment = "Employment (payrolls, unemployment, JOLTS, wages)",
  macro      = "Additional Macro (sentiment, housing, dollar)"
)

ALL_CATEGORIES <- names(CATEGORIES)


# =============================================================================
# DEPENDENCY-INJECTED LOADERS
# =============================================================================
# These wrappers need access to the partially-loaded data_dict.

.etl_load_stockdata <- function(data_dict,
                                start_date = ETL_START_DATE,
                                end_date   = ETL_END_DATE) {
  cw <- data_dict[["culturewardata"]]
  if (is.null(cw)) {
    warning("Skipping stockdata: culturewardata not loaded")
    return(NULL)
  }
  tickers <- load_culture_war_companies(cw)
  get_stock_data(tickers, start_date = start_date, end_date = end_date)
}

.etl_load_controlcompanies <- function(data_dict) {
  cw <- data_dict[["culturewardata"]]
  if (is.null(cw)) {
    warning("Skipping controlcompanies: culturewardata not loaded")
    return(NULL)
  }
  build_control_companies(cw)
}

.etl_load_political_exposure <- function(data_dict,
                                          start_date = ETL_START_DATE,
                                          end_date   = ETL_END_DATE) {
  cw <- data_dict[["culturewardata"]]
  tickers <- if (!is.null(cw)) load_culture_war_companies(cw) else NULL
  load_political_exposure(tickers = tickers,
                          start_date = start_date, end_date = end_date)
}

# Set of loader names that require data_dict as first argument
.DATA_DICT_LOADERS <- c(
  ".etl_load_stockdata",
  ".etl_load_controlcompanies",
  ".etl_load_political_exposure"
)


# =============================================================================
# DEPENDENCY RESOLUTION
# =============================================================================

#' Resolve dependencies: topological sort ensuring parents load first.
#'
#' @param load_keys Character vector of dataset keys to load.
#' @return Character vector in dependency-resolved order.
.resolve_dependencies <- function(load_keys) {
  resolved <- character(0)
  seen     <- character(0)

  add_key <- function(key) {
    if (key %in% seen) return()
    entry <- DATA_DICTIONARY[[key]]
    if (is.null(entry)) return()
    for (dep in entry$depends_on) {
      add_key(dep)
    }
    seen <<- c(seen, key)
    resolved <<- c(resolved, key)
  }

  for (key in load_keys) {
    add_key(key)
  }
  resolved
}


# =============================================================================
# SINGLE DATASET LOADER
# =============================================================================

#' Load a single dataset entry from the data dictionary.
#'
#' @param key       Dataset key name.
#' @param entry     Data dictionary entry (list).
#' @param data_dict Current partially-loaded data dictionary.
#' @param start_date Override start date (or NULL).
#' @param end_date   Override end date (or NULL).
#' @return The loaded dataset (tibble, list, or NULL).
.load_single <- function(key, entry, data_dict,
                          start_date = NULL, end_date = NULL) {
  loader_name <- entry$loader
  args <- entry$args

  # Override date range if specified
  if (!is.null(start_date) && "start_date" %in% names(args)) {
    args$start_date <- start_date
  }
  if (!is.null(end_date) && "end_date" %in% names(args)) {
    args$end_date <- end_date
  }

  # Dependency-injected loaders: prepend data_dict
  if (loader_name %in% .DATA_DICT_LOADERS) {
    fn <- get(loader_name, envir = globalenv())
    args <- c(list(data_dict = data_dict), args)
    return(do.call(fn, args))
  }

  # Standard loaders
  fn <- get(loader_name, envir = globalenv())
  do.call(fn, args)
}


# =============================================================================
# ETL PIPELINE
# =============================================================================

#' Run the ETL pipeline to load datasets.
#'
#' @param categories Character vector of category names to load.
#'        Options: "market", "political", "inflation", "rates",
#'        "production", "money", "gdp", "employment", "macro".
#'        Defaults to all categories.
#' @param keys       Character vector of specific dataset keys (overrides
#'        categories).
#' @param start_date Override default start date ("2000-01-01").
#' @param end_date   Override default end date ("2025-12-31").
#' @param do_clean   If TRUE, apply cleaning after loading.
#' @param verbose    If TRUE, print progress and summary.
#' @return A named list of loaded (and optionally cleaned) datasets.
run_etl <- function(categories = NULL,
                    keys       = NULL,
                    start_date = NULL,
                    end_date   = NULL,
                    do_clean   = TRUE,
                    verbose    = TRUE) {

  if (is.null(categories) && is.null(keys)) {
    categories <- ALL_CATEGORIES
  }

  # Determine which keys to load
  if (!is.null(keys)) {
    valid_keys <- intersect(keys, names(DATA_DICTIONARY))
    invalid    <- setdiff(keys, names(DATA_DICTIONARY))
    if (length(invalid) > 0) {
      warning("Unknown keys (skipped): ", paste(invalid, collapse = ", "))
    }
    load_keys <- valid_keys
  } else {
    load_keys <- names(DATA_DICTIONARY)[
      sapply(DATA_DICTIONARY, function(e) e$category %in% categories)
    ]
  }

  # Resolve dependencies
  resolved <- .resolve_dependencies(load_keys)
  extra    <- setdiff(resolved, load_keys)
  if (length(extra) > 0 && verbose) {
    message("  Auto-added dependencies: ", paste(extra, collapse = ", "))
  }

  if (verbose) {
    message(strrep("=", 60))
    message("  Signals & Systems ETL Pipeline (R)")
    message(strrep("=", 60))
    message("  Datasets to load: ", length(resolved))
    if (!is.null(categories) && is.null(keys)) {
      message("  Categories: ", paste(categories, collapse = ", "))
    }
    message("  Date range: ",
            start_date %||% ETL_START_DATE, " to ",
            end_date %||% ETL_END_DATE)
    message("  Clean data: ", do_clean)
    message(strrep("=", 60))
  }

  data_dict <- list()
  timings   <- list()
  errors    <- list()

  for (key in resolved) {
    entry <- DATA_DICTIONARY[[key]]
    if (verbose) message("[LOADING] ", key, " - ", entry$description)

    t0 <- proc.time()["elapsed"]

    tryCatch({
      result <- .load_single(key, entry, data_dict, start_date, end_date)
      data_dict[[key]] <- result
      elapsed <- proc.time()["elapsed"] - t0
      timings[[key]] <- elapsed

      status <- if (!is.null(result)) "OK" else "EMPTY"
      if (verbose) message("  [", status, "] ", key, " (", round(elapsed, 1), "s)")

    }, error = function(e) {
      data_dict[[key]] <<- NULL
      elapsed <- proc.time()["elapsed"] - t0
      timings[[key]] <<- elapsed
      errors[[key]] <<- conditionMessage(e)
      message("  [FAIL] ", key, ": ", conditionMessage(e),
              " (", round(elapsed, 1), "s)")
    })
  }

  # Apply cleaning
  if (do_clean) {
    if (verbose) message("Applying data cleaning...")
    data_dict <- clean_all_data(data_dict, verbose = verbose)
  }

  # Print summary
  if (verbose) {
    .print_summary(data_dict, timings, errors)
  }

  data_dict
}


# =============================================================================
# SUMMARY REPORTING
# =============================================================================

#' Get a human-readable shape description for a dataset.
#'
#' @param data A dataset (tibble, list, or NULL).
#' @return Character string describing the shape.
.get_shape <- function(data) {
  if (is.null(data)) return("NULL")
  if (is.data.frame(data)) {
    return(paste0(format(nrow(data), big.mark = ","), " rows x ", ncol(data), " cols"))
  }
  if (is.list(data)) {
    n_keys <- length(data)
    combined <- data[["combined"]]
    if (!is.null(combined) && is.data.frame(combined)) {
      return(paste0("list(", n_keys, " keys, combined: ",
                    format(nrow(combined), big.mark = ","), " x ",
                    ncol(combined), ")"))
    }
    return(paste0("list(", n_keys, " keys)"))
  }
  class(data)[1]
}


#' Print ETL summary report.
.print_summary <- function(data_dict, timings, errors) {
  total_time <- sum(unlist(timings))

  message("")
  message(strrep("=", 60))
  message("  ETL Summary")
  message(strrep("=", 60))

  loaded <- names(data_dict)[!sapply(data_dict, is.null)]
  failed <- names(data_dict)[sapply(data_dict, is.null)]

  message("  Loaded:  ", length(loaded), " datasets")
  message("  Failed:  ", length(failed), " datasets")
  message("  Total time: ", round(total_time, 1), "s")
  message("")

  for (cat_key in names(CATEGORIES)) {
    cat_desc <- CATEGORIES[[cat_key]]
    cat_keys <- names(DATA_DICTIONARY)[
      sapply(DATA_DICTIONARY, function(e) e$category == cat_key)
    ]
    cat_keys <- intersect(cat_keys, names(data_dict))
    if (length(cat_keys) == 0) next

    message("  --- ", cat_desc, " ---")
    for (key in cat_keys) {
      val <- data_dict[[key]]
      t   <- timings[[key]] %||% 0
      if (is.null(val)) {
        err <- errors[[key]] %||% "unknown"
        message(sprintf("    %-30s FAILED (%s)", key, err))
      } else {
        shape <- .get_shape(val)
        message(sprintf("    %-30s %s  (%.1fs)", key, shape, t))
      }
    }
    message("")
  }

  if (length(errors) > 0) {
    message("  --- Errors ---")
    for (key in names(errors)) {
      message("    ", key, ": ", errors[[key]])
    }
  }
  message(strrep("=", 60))
}


#' Print a quick summary of a loaded data dictionary.
#'
#' @param data_dict Named list of datasets from run_etl().
summarize_data <- function(data_dict) {
  message("")
  message(strrep("=", 70))
  message("  Dataset Summary")
  message(strrep("=", 70))

  for (key in names(data_dict)) {
    val  <- data_dict[[key]]
    entry <- DATA_DICTIONARY[[key]]
    cat  <- if (!is.null(entry)) entry$category else "?"

    if (is.null(val)) {
      message(sprintf("  [%10s] %-30s -- not loaded --", cat, key))
    } else if (is.data.frame(val)) {
      message(sprintf("  [%10s] %-30s %6s rows x %3d cols",
                      cat, key,
                      format(nrow(val), big.mark = ","),
                      ncol(val)))
    } else if (is.list(val)) {
      combined <- val[["combined"]]
      if (!is.null(combined) && is.data.frame(combined)) {
        message(sprintf("  [%10s] %-30s %6s rows x %3d cols (combined)",
                        cat, key,
                        format(nrow(combined), big.mark = ","),
                        ncol(combined)))
      } else {
        message(sprintf("  [%10s] %-30s list with %d keys",
                        cat, key, length(val)))
      }
    } else {
      message(sprintf("  [%10s] %-30s %s", cat, key, class(val)[1]))
    }
  }
  message(strrep("=", 70))
}


#' List all available datasets and their descriptions.
list_datasets <- function() {
  message("")
  message(strrep("=", 70))
  message("  Available Datasets")
  message(strrep("=", 70))

  for (cat_key in names(CATEGORIES)) {
    cat_desc <- CATEGORIES[[cat_key]]
    message("")
    message("  --- ", cat_desc, " ---")

    for (key in names(DATA_DICTIONARY)) {
      entry <- DATA_DICTIONARY[[key]]
      if (entry$category != cat_key) next
      deps <- entry$depends_on
      dep_str <- if (length(deps) > 0) {
        paste0(" (requires: ", paste(deps, collapse = ", "), ")")
      } else ""
      message(sprintf("    %-30s %s%s", key, entry$description, dep_str))
    }
  }
  message("")
  message(strrep("=", 70))
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

#' Load only macroeconomic data (no market/stock data).
#'
#' @param do_clean If TRUE, apply cleaning.
#' @return Named list of macro datasets.
load_macro_only <- function(do_clean = TRUE) {
  run_etl(
    categories = c("inflation", "rates", "production",
                    "money", "gdp", "employment", "macro"),
    do_clean   = do_clean
  )
}

#' Load only market data (stocks, VIX, Fama-French).
#'
#' @param do_clean If TRUE, apply cleaning.
#' @return Named list of market datasets.
load_market_only <- function(do_clean = TRUE) {
  run_etl(categories = "market", do_clean = do_clean)
}

#' Load only political data.
#'
#' @param do_clean If TRUE, apply cleaning.
#' @return Named list of political datasets.
load_political_only <- function(do_clean = TRUE) {
  run_etl(categories = "political", do_clean = do_clean)
}


# =============================================================================
# MAIN (when sourced with Rscript)
# =============================================================================

message("")
message("etl.R loaded successfully.")
message("  run_etl()          — Run the full ETL pipeline")
message("  summarize_data(d)  — Summarize a loaded data dictionary")
message("  list_datasets()    — Show all available datasets")
message("  load_macro_only()  — Load only macroeconomic data")
message("  load_market_only() — Load only market data")
