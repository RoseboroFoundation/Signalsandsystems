# =============================================================================
# clean.R — Data Cleaning and Aggregation for Culture War Companies Research
#
# R port of the Python `clean` package (config, market_data, fred_loaders,
# orchestration, orthogonalize, political_events, political_exposure).
#
# Dependencies:
#   install.packages(c("tidyverse", "fredr", "quantmod", "tidyquant",
#                      "httr", "jsonlite", "readr", "zoo", "lubridate"))
#
# Usage:
#   source("clean.R")
#   cw <- import_culture_war_data("Culture_War_Companies_160_fullmeta.csv")
#   tickers <- load_culture_war_companies(cw)
#   stock <- get_stock_data(tickers[1:5], "2020-01-01", "2025-12-31")
# =============================================================================

library(tidyverse)
library(fredr)
library(quantmod)
library(tidyquant)
library(httr)
library(jsonlite)
library(readr)
library(zoo)
library(lubridate)

# =============================================================================
# CONFIGURATION
# =============================================================================

#' Validate and set the FRED API key.
#'
#' Reads FRED_API_KEY from .Renviron, environment variable, or a .env file
#' in the project root. Raises an error if not found.
#'
#' @return Invisibly returns the API key string.
validate_fred_api_key <- function() {

  key <- Sys.getenv("FRED_API_KEY", unset = "")


  # Try loading from .env if not already set

  if (nchar(key) == 0L) {
    env_file <- file.path(getwd(), ".env")
    if (file.exists(env_file)) {
      lines <- readLines(env_file, warn = FALSE)
      fred_line <- grep("^FRED_API_KEY=", lines, value = TRUE)
      if (length(fred_line) > 0L) {
        key <- sub("^FRED_API_KEY=", "", fred_line[1])
        key <- trimws(key, whitespace = "[\"' \t]")
        Sys.setenv(FRED_API_KEY = key)
      }
    }
  }

  if (nchar(key) == 0L) {
    stop(
      "FRED API key not found. Set FRED_API_KEY in .Renviron or .env.\n",
      "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html",
      call. = FALSE
    )
  }

  fredr_set_key(key)
  invisible(key)
}

# Module-level defaults
START_DATE <- "2000-01-01"
END_DATE   <- "2025-12-31"
CACHE_PATH <- "./data/fred"
VIX_OUTPUT_FILE <- "vix_data_2000_2025.csv"


# =============================================================================
# FRED HELPER
# =============================================================================

#' Download multiple FRED series into a single tibble (wide format).
#'
#' @param series_dict Named character vector: display_name -> FRED series ID.
#' @param start_date Start date string (YYYY-MM-DD).
#' @param end_date   End date string (YYYY-MM-DD).
#' @return A tibble with a `date` column and one column per series.
download_fred_series <- function(series_dict,
                                 start_date = START_DATE,
                                 end_date   = END_DATE) {
  validate_fred_api_key()

  all_data <- list()
  for (nm in names(series_dict)) {
    sid <- series_dict[[nm]]
    tryCatch({
      message("  Downloading ", nm, " (", sid, ")...")
      raw <- fredr(
        series_id        = sid,
        observation_start = as.Date(start_date),
        observation_end   = as.Date(end_date)
      )
      if (nrow(raw) > 0) {
        all_data[[nm]] <- tibble(date = raw$date, !!nm := raw$value)
      }
    }, error = function(e) {
      warning("Could not download ", nm, " (", sid, "): ", conditionMessage(e))
    })
  }

  if (length(all_data) == 0L) return(tibble(date = as.Date(character(0))))

  # Full outer join on date
  result <- all_data[[1]]
  if (length(all_data) > 1L) {
    for (i in 2:length(all_data)) {
      result <- full_join(result, all_data[[i]], by = "date")
    }
  }
  result <- arrange(result, date)
  result
}


# =============================================================================
# CULTURE WAR DATA
# =============================================================================

#' Import and clean the Culture War Companies dataset from CSV.
#'
#' @param file_path Path to the CSV file.
#' @return A tibble with parsed Event Date column.
import_culture_war_data <- function(file_path) {
  if (!file.exists(file_path)) {
    # Try relative to project root
    alt <- file.path(dirname(getwd()), file_path)
    if (file.exists(alt)) file_path <- alt
  }
  df <- read_csv(file_path, show_col_types = FALSE)
  if ("Event Date" %in% names(df)) {
    df <- df %>% mutate(`Event Date` = as.Date(`Event Date`, format = "%Y-%m-%d"))
  }
  df
}


# =============================================================================
# MARKET DATA — Stock Prices via quantmod / tidyquant
# =============================================================================

#' Download stock data for a vector of tickers from Yahoo Finance.
#'
#' @param tickers   Character vector of ticker symbols.
#' @param start_date Start date (YYYY-MM-DD).
#' @param end_date   End date (YYYY-MM-DD).
#' @return A named list of tibbles, one per ticker (columns: Ticker, Date,
#'         Open, High, Low, Close, Volume, Adjusted).
get_stock_data <- function(tickers,
                           start_date = "2000-01-01",
                           end_date   = "2025-12-31") {
  stock_data   <- list()
  failed       <- character(0)

  for (ticker in tickers) {
    tryCatch({
      message("Downloading data for ", ticker, "...")
      raw <- tq_get(
        ticker,
        get  = "stock.prices",
        from = start_date,
        to   = end_date
      )

      if (!is.null(raw) && nrow(raw) > 0) {
        raw <- raw %>%
          mutate(Ticker = ticker) %>%
          select(Ticker, Date = date, Open = open, High = high,
                 Low = low, Close = close, Volume = volume,
                 Adjusted = adjusted)
        stock_data[[ticker]] <- raw
        message("  Successfully downloaded ", nrow(raw), " rows for ", ticker)
      } else {
        failed <- c(failed, ticker)
        message("  No data found for ", ticker)
      }
    }, error = function(e) {
      failed <<- c(failed, ticker)
      warning("Error downloading ", ticker, ": ", conditionMessage(e))
    })
  }

  if (length(failed) > 0) {
    warning("Failed to download data for: ", paste(failed, collapse = ", "))
  }
  stock_data
}


#' Download VIX (VIXCLS) from FRED and optionally save to CSV.
#'
#' @param save_csv Logical; if TRUE, writes to VIX_OUTPUT_FILE.
#' @return A tibble with columns `date` and `vix`.
download_vix_data <- function(save_csv = TRUE) {
  validate_fred_api_key()

  message("Downloading VIX data from FRED...")
  vix_raw <- fredr(
    series_id        = "VIXCLS",
    observation_start = as.Date(START_DATE),
    observation_end   = as.Date(END_DATE)
  )

  vix_df <- tibble(date = vix_raw$date, vix = vix_raw$value) %>%
    drop_na()

  if (save_csv) {
    write_csv(vix_df, VIX_OUTPUT_FILE)
    message("Saved VIX data to ", VIX_OUTPUT_FILE)
  }

  message("VIX: ", nrow(vix_df), " observations, ",
          min(vix_df$date), " to ", max(vix_df$date))
  message("  Mean = ", round(mean(vix_df$vix), 2),
          ", Max = ", round(max(vix_df$vix), 2))
  vix_df
}


#' Download Fama-French factor data from the Ken French data library.
#'
#' Uses tidyquant's tq_get with "Fama-French" source.
#'
#' @param start_date Start date string.
#' @param frequency  "daily" or "monthly".
#' @param output_dir Directory to save CSVs (NULL to skip saving).
#' @return A named list with elements FF3, FF5, MOM (tibbles).
download_fama_french_factors <- function(start_date = "1926-07-01",
                                         end_date   = NULL,
                                         frequency  = "daily",
                                         output_dir = NULL) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  ff3_name <- if (frequency == "daily") {
    "F-F_Research_Data_Factors_daily"
  } else {
    "F-F_Research_Data_Factors"
  }
  ff5_name <- if (frequency == "daily") {
    "F-F_Research_Data_5_Factors_2x3_daily"
  } else {
    "F-F_Research_Data_5_Factors_2x3"
  }
  mom_name <- if (frequency == "daily") {
    "F-F_Momentum_Factor_daily"
  } else {
    "F-F_Momentum_Factor"
  }

  results <- list()
  tryCatch({
    message("Downloading Fama-French 3-Factor Model...")
    results[["FF3"]] <- tq_get(ff3_name, get = "Fama-French")
    message("Downloading Fama-French 5-Factor Model...")
    results[["FF5"]] <- tq_get(ff5_name, get = "Fama-French")
    message("Downloading Momentum Factor...")
    results[["MOM"]] <- tq_get(mom_name, get = "Fama-French")

    if (!is.null(output_dir)) {
      dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
      for (nm in names(results)) {
        fp <- file.path(output_dir, paste0(nm, "_", frequency, ".csv"))
        write_csv(results[[nm]], fp)
        message("Saved ", nm, " to ", fp)
      }
    }
    message("Fama-French download complete.")
  }, error = function(e) {
    warning("Error downloading Fama-French data: ", conditionMessage(e))
    return(NULL)
  })
  results
}


#' Download Fama-French industry portfolio returns.
#'
#' @param num_industries Number of industries (10, 12, 17, 30, 48, 49).
#' @param start_date     Start date.
#' @param end_date       End date.
#' @param frequency      "daily" or "monthly".
#' @param output_dir     Directory to save CSV (NULL to skip).
#' @return A tibble of industry portfolio returns.
download_industry_portfolios <- function(num_industries = 10,
                                          start_date     = "1926-07-01",
                                          end_date       = NULL,
                                          frequency      = "daily",
                                          output_dir     = NULL) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  suffix <- if (frequency == "daily") "_daily" else ""
  ds_name <- paste0(num_industries, "_Industry_Portfolios", suffix)

  tryCatch({
    message("Downloading ", num_industries, " Industry Portfolios...")
    df <- tq_get(ds_name, get = "Fama-French")

    if (!is.null(output_dir)) {
      dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
      fp <- file.path(output_dir,
                      paste0("Industry_", num_industries, "_", frequency, ".csv"))
      write_csv(df, fp)
      message("Saved to ", fp)
    }
    df
  }, error = function(e) {
    warning("Error downloading industry portfolios: ", conditionMessage(e))
    NULL
  })
}


# =============================================================================
# FRED MACRO LOADERS — Inflation
# =============================================================================

#' Load core inflation data from FRED (CPI, PCE, PPI, GDP Deflator).
#'
#' @param start_date Start date.
#' @param end_date   End date (defaults to today).
#' @param cache_path Directory for caching (unused in R version — relies
#'        on fredr caching or user-level caching).
#' @return A named list with `raw`, `yoy`, `mom`, and `combined` tibbles.
load_inflation_data <- function(start_date = START_DATE,
                                end_date   = NULL,
                                cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  series <- c(
    CPI          = "CPIAUCSL",
    Core_CPI     = "CPILFESL",
    PCE          = "PCEPI",
    Core_PCE     = "PCEPILFE",
    PPI          = "PPIFIS",
    GDP_Deflator = "GDPDEF"
  )

  message("Downloading inflation data from FRED...")
  raw <- download_fred_series(series, start_date, end_date)

  # Year-over-year pct change (12-period lag for monthly data)
  yoy <- raw %>%
    mutate(across(-date, ~ (. / lag(., 12) - 1) * 100,
                  .names = "{.col}_YoY"))
  yoy <- yoy %>% select(date, ends_with("_YoY"))

  # Month-over-month annualized pct change
  mom <- raw %>%
    mutate(across(-date, ~ (. / lag(., 1) - 1) * 100 * 12,
                  .names = "{.col}_MoM"))
  mom <- mom %>% select(date, ends_with("_MoM"))

  combined <- raw %>%
    left_join(yoy, by = "date") %>%
    left_join(mom, by = "date")

  message("Inflation data: ", nrow(raw), " observations, ",
          ncol(raw) - 1, " series")

  list(raw = raw, yoy = yoy, mom = mom, combined = combined)
}


#' Load inflation expectations data from FRED.
#'
#' Includes breakeven inflation, survey expectations, and Fed measures.
#'
#' @inheritParams load_inflation_data
#' @return A named list with `breakeven`, `survey`, `fed_measures`, `combined`.
load_inflation_expectations_data <- function(start_date = START_DATE,
                                              end_date   = NULL,
                                              cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  breakeven_series <- c(
    Breakeven_5Y   = "T5YIE",
    Breakeven_10Y  = "T10YIE",
    Breakeven_5Y5Y = "T5YIFR"
  )
  survey_series <- c(
    UMich_Inflation_1Y = "MICH"
  )
  fed_series <- c(
    Trimmed_Mean_PCE   = "PCETRIM12M159SFRBDAL",
    Sticky_Price_CPI   = "CORESTICKM159SFRBATL",
    Flexible_Price_CPI = "FLEXCPIM159SFRBATL",
    Median_CPI         = "MEDCPIM158SFRBCLE"
  )

  breakeven_df <- download_fred_series(breakeven_series, start_date, end_date)
  survey_df    <- download_fred_series(survey_series, start_date, end_date)
  fed_df       <- download_fred_series(fed_series, start_date, end_date)

  combined <- breakeven_df %>%
    full_join(survey_df, by = "date") %>%
    full_join(fed_df, by = "date") %>%
    arrange(date)

  list(breakeven = breakeven_df, survey = survey_df,
       fed_measures = fed_df, combined = combined)
}


#' Load comprehensive inflation data (all measures combined).
#'
#' @inheritParams load_inflation_data
#' @return A named list with `core`, `expectations`, `components`, `combined`.
load_comprehensive_inflation_data <- function(start_date = START_DATE,
                                               end_date   = NULL,
                                               cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  core_data         <- load_inflation_data(start_date, end_date, cache_path)
  expectations_data <- load_inflation_expectations_data(start_date, end_date, cache_path)

  component_series <- c(
    CPI_Food           = "CPIUFDSL",
    CPI_Energy         = "CPIENGSL",
    CPI_Shelter        = "CUSR0000SAH1",
    CPI_Medical        = "CPIMEDSL",
    CPI_Transportation = "CPITRNSL",
    CPI_Apparel        = "CPIAPPSL",
    CPI_Services       = "CUSR0000SAS",
    Import_Prices      = "IR",
    Export_Prices      = "IQ"
  )

  components_df <- download_fred_series(component_series, start_date, end_date)
  components_yoy <- components_df %>%
    mutate(across(-date, ~ (. / lag(., 12) - 1) * 100,
                  .names = "{.col}_YoY"))
  components_yoy <- components_yoy %>% select(date, ends_with("_YoY"))

  combined <- core_data$combined %>%
    full_join(expectations_data$combined, by = "date") %>%
    full_join(components_yoy, by = "date") %>%
    arrange(date)

  list(core = core_data, expectations = expectations_data,
       components = list(raw = components_df, yoy = components_yoy),
       combined = combined)
}


# =============================================================================
# FRED MACRO LOADERS — Interest Rates
# =============================================================================

#' Load Treasury yield curve data from FRED (1M through 30Y plus TIPS).
#'
#' @inheritParams load_inflation_data
#' @return Named list with `nominal`, `real`, `combined` tibbles.
load_treasury_yields <- function(start_date = START_DATE,
                                  end_date   = NULL,
                                  cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  nominal <- c(
    Treasury_1M  = "DGS1MO", Treasury_3M  = "DGS3MO",
    Treasury_6M  = "DGS6MO", Treasury_1Y  = "DGS1",
    Treasury_2Y  = "DGS2",   Treasury_3Y  = "DGS3",
    Treasury_5Y  = "DGS5",   Treasury_7Y  = "DGS7",
    Treasury_10Y = "DGS10",  Treasury_20Y = "DGS20",
    Treasury_30Y = "DGS30"
  )
  tips <- c(
    TIPS_5Y  = "DFII5",  TIPS_10Y = "DFII10",
    TIPS_20Y = "DFII20", TIPS_30Y = "DFII30"
  )

  message("Downloading Treasury yields...")
  nominal_df <- download_fred_series(nominal, start_date, end_date)
  tips_df    <- download_fred_series(tips, start_date, end_date)

  combined <- full_join(nominal_df, tips_df, by = "date") %>% arrange(date)
  list(nominal = nominal_df, real = tips_df, combined = combined)
}


#' Load Federal Reserve policy rates from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `fed_funds`, `money_market`, `combined`.
load_policy_rates <- function(start_date = START_DATE,
                               end_date   = NULL,
                               cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  fed <- c(
    Fed_Funds_Effective    = "DFF",
    Fed_Funds_Target_Upper = "DFEDTARU",
    Fed_Funds_Target_Lower = "DFEDTARL",
    Discount_Rate          = "INTDSRUSM193N"
  )
  mm <- c(
    SOFR                   = "SOFR",
    Prime_Rate             = "DPRIME",
    Overnight_Bank_Funding = "OBFR",
    EFFR                   = "EFFR"
  )

  fed_df <- download_fred_series(fed, start_date, end_date)
  mm_df  <- download_fred_series(mm, start_date, end_date)

  combined <- full_join(fed_df, mm_df, by = "date") %>% arrange(date)
  list(fed_funds = fed_df, money_market = mm_df, combined = combined)
}


#' Load credit spreads and corporate bond yields from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `corporate`, `spreads`, `mortgage`, `combined`.
load_credit_spreads <- function(start_date = START_DATE,
                                 end_date   = NULL,
                                 cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  corporate <- c(
    Moodys_AAA         = "AAA",
    Moodys_BAA         = "BAA",
    ICE_BofA_HighYield = "BAMLH0A0HYM2EY"
  )
  spreads <- c(
    BAA_10Y_Spread = "BAA10Y",
    AAA_10Y_Spread = "AAA10Y",
    IG_Spread      = "BAMLC0A0CM",
    HY_Spread      = "BAMLH0A0HYM2",
    TED_Spread     = "TEDRATE"
  )
  mortgage <- c(
    Mortgage_30Y = "MORTGAGE30US",
    Mortgage_15Y = "MORTGAGE15US"
  )

  corp_df     <- download_fred_series(corporate, start_date, end_date)
  spread_df   <- download_fred_series(spreads, start_date, end_date)
  mortgage_df <- download_fred_series(mortgage, start_date, end_date)

  combined <- corp_df %>%
    full_join(spread_df, by = "date") %>%
    full_join(mortgage_df, by = "date") %>%
    arrange(date)

  list(corporate = corp_df, spreads = spread_df,
       mortgage = mortgage_df, combined = combined)
}


#' Calculate yield curve metrics from Treasury data.
#'
#' @param treasury_data Output from load_treasury_yields().
#' @return A tibble with slope, curvature, and inversion indicators.
calculate_yield_curve_metrics <- function(treasury_data) {
  if (is.null(treasury_data) || is.null(treasury_data$nominal)) {
    message("Treasury data not available")
    return(NULL)
  }
  nom <- treasury_data$nominal
  metrics <- tibble(date = nom$date)

  if (all(c("Treasury_10Y", "Treasury_2Y") %in% names(nom))) {
    metrics$Slope_10Y_2Y <- nom$Treasury_10Y - nom$Treasury_2Y
  }
  if (all(c("Treasury_10Y", "Treasury_3M") %in% names(nom))) {
    metrics$Slope_10Y_3M <- nom$Treasury_10Y - nom$Treasury_3M
  }
  if (all(c("Treasury_2Y", "Treasury_5Y", "Treasury_10Y") %in% names(nom))) {
    metrics$Curvature_2_5_10 <- 2 * nom$Treasury_5Y - nom$Treasury_2Y - nom$Treasury_10Y
  }
  if ("Slope_10Y_2Y" %in% names(metrics)) {
    metrics$Inverted_10Y_2Y <- as.integer(metrics$Slope_10Y_2Y < 0)
  }
  if ("Slope_10Y_3M" %in% names(metrics)) {
    metrics$Inverted_10Y_3M <- as.integer(metrics$Slope_10Y_3M < 0)
  }
  if (all(c("Treasury_2Y", "Treasury_5Y", "Treasury_10Y") %in% names(nom))) {
    metrics$Curve_Level <- (nom$Treasury_2Y + nom$Treasury_5Y + nom$Treasury_10Y) / 3
  }
  metrics
}


#' Load comprehensive rates data (Treasury + policy + credit + curve metrics).
#'
#' @inheritParams load_inflation_data
#' @return Named list with `treasury`, `policy`, `credit`,
#'         `curve_metrics`, `combined`.
load_comprehensive_rates_data <- function(start_date = START_DATE,
                                           end_date   = NULL,
                                           cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  treasury <- load_treasury_yields(start_date, end_date, cache_path)
  policy   <- load_policy_rates(start_date, end_date, cache_path)
  credit   <- load_credit_spreads(start_date, end_date, cache_path)
  metrics  <- calculate_yield_curve_metrics(treasury)

  combined <- treasury$combined %>%
    full_join(policy$combined, by = "date") %>%
    full_join(credit$combined, by = "date")
  if (!is.null(metrics)) {
    combined <- full_join(combined, metrics, by = "date")
  }
  combined <- combined %>%
    arrange(date) %>%
    select(where(~ !duplicated(cur_column())))

  list(treasury = treasury, policy = policy, credit = credit,
       curve_metrics = metrics, combined = combined)
}


# =============================================================================
# FRED MACRO LOADERS — Industrial Production
# =============================================================================

#' Load Industrial Production data from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `total`, `sectors`, `capacity`, `combined`.
load_industrial_production_data <- function(start_date = START_DATE,
                                             end_date   = NULL,
                                             cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  total <- c(IP_Total = "INDPRO")
  sectors <- c(
    IP_Manufacturing = "IPMAN",
    IP_Mining        = "IPMINE",
    IP_Utilities     = "IPUTIL"
  )
  capacity <- c(
    CapUtil_Total         = "TCU",
    CapUtil_Manufacturing = "MCUMFN"
  )

  total_df    <- download_fred_series(total, start_date, end_date)
  sector_df   <- download_fred_series(sectors, start_date, end_date)
  capacity_df <- download_fred_series(capacity, start_date, end_date)

  combined <- total_df %>%
    full_join(sector_df, by = "date") %>%
    full_join(capacity_df, by = "date") %>%
    arrange(date)

  list(total = total_df, sectors = sector_df,
       capacity = capacity_df, combined = combined)
}


#' Load IP growth rates (YoY, MoM) and diffusion indices.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `raw`, `yoy`, `mom`, `combined`.
load_ip_growth_rates <- function(start_date = START_DATE,
                                  end_date   = NULL,
                                  cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  ip_series <- c(IP = "INDPRO", IP_Manufacturing = "IPMAN")
  ip_df <- download_fred_series(ip_series, start_date, end_date)

  yoy <- ip_df %>%
    mutate(across(-date, ~ (. / lag(., 12) - 1) * 100,
                  .names = "{.col}_YoY"))
  yoy <- yoy %>% select(date, ends_with("_YoY"))

  mom <- ip_df %>%
    mutate(across(-date, ~ (. / lag(., 1) - 1) * 100,
                  .names = "{.col}_MoM"))
  mom <- mom %>% select(date, ends_with("_MoM"))

  combined <- yoy %>% full_join(mom, by = "date") %>% arrange(date)
  list(raw = ip_df, yoy = yoy, mom = mom, combined = combined)
}


#' Load comprehensive IP data.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `production`, `growth`, `combined`.
load_comprehensive_ip_data <- function(start_date = START_DATE,
                                        end_date   = NULL,
                                        cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  prod   <- load_industrial_production_data(start_date, end_date, cache_path)
  growth <- load_ip_growth_rates(start_date, end_date, cache_path)

  combined <- prod$combined %>%
    full_join(growth$combined, by = "date") %>%
    arrange(date)

  list(production = prod, growth = growth, combined = combined)
}


# =============================================================================
# FRED MACRO LOADERS — Money Supply
# =============================================================================

#' Load M1, M2, monetary base, and components from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `aggregates`, `components`, `combined`.
load_money_supply_data <- function(start_date = START_DATE,
                                    end_date   = NULL,
                                    cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  agg <- c(
    M1            = "M1SL",
    M2            = "M2SL",
    Monetary_Base = "BOGMBASE"
  )
  comp <- c(
    Currency_Circulation = "CURRSL",
    Demand_Deposits      = "DEMDEPSL",
    Savings_Deposits     = "SAVINGSL"
  )

  agg_df  <- download_fred_series(agg, start_date, end_date)
  comp_df <- download_fred_series(comp, start_date, end_date)

  combined <- full_join(agg_df, comp_df, by = "date") %>% arrange(date)
  list(aggregates = agg_df, components = comp_df, combined = combined)
}


#' Load M1 and M2 velocity from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `velocity`, `combined`.
load_money_velocity_data <- function(start_date = START_DATE,
                                      end_date   = NULL,
                                      cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  vel <- c(M1_Velocity = "M1V", M2_Velocity = "M2V")
  vel_df <- download_fred_series(vel, start_date, end_date)
  list(velocity = vel_df, combined = vel_df)
}


#' Load Federal Reserve Balance Sheet data from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `assets`, `reserves`, `combined`.
load_fed_balance_sheet_data <- function(start_date = START_DATE,
                                         end_date   = NULL,
                                         cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  assets <- c(
    Fed_Total_Assets     = "WALCL",
    Fed_Treasury_Holdings = "TREAST",
    Fed_MBS_Holdings     = "WSHOMCB"
  )
  reserves <- c(
    Reserve_Balances = "WRESBAL",
    Total_Reserves   = "TOTRESNS"
  )

  asset_df   <- download_fred_series(assets, start_date, end_date)
  reserve_df <- download_fred_series(reserves, start_date, end_date)

  combined <- full_join(asset_df, reserve_df, by = "date") %>% arrange(date)
  list(assets = asset_df, reserves = reserve_df, combined = combined)
}


#' Load comprehensive M2 data (supply + velocity + Fed balance sheet + growth).
#'
#' @inheritParams load_inflation_data
#' @return Named list with `money_supply`, `velocity`, `fed_balance_sheet`,
#'         `growth_rates`, `combined`.
load_comprehensive_m2_data <- function(start_date = START_DATE,
                                        end_date   = NULL,
                                        cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  ms  <- load_money_supply_data(start_date, end_date, cache_path)
  vel <- load_money_velocity_data(start_date, end_date, cache_path)
  fed <- load_fed_balance_sheet_data(start_date, end_date, cache_path)

  # Calculate growth rates
  growth <- ms$aggregates %>%
    mutate(
      M2_YoY = (M2 / lag(M2, 12) - 1) * 100,
      M1_YoY = (M1 / lag(M1, 12) - 1) * 100,
      M2_MoM = (M2 / lag(M2, 1) - 1) * 100 * 12
    ) %>%
    select(date, M2_YoY, M1_YoY, M2_MoM)

  combined <- ms$combined %>%
    full_join(vel$combined, by = "date") %>%
    full_join(fed$combined, by = "date") %>%
    full_join(growth, by = "date") %>%
    arrange(date)

  list(money_supply = ms, velocity = vel, fed_balance_sheet = fed,
       growth_rates = growth, combined = combined)
}


# =============================================================================
# FRED MACRO LOADERS — GDP
# =============================================================================

#' Load headline GDP data from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `headline`, `growth`, `per_capita`, `combined`.
load_gdp_data <- function(start_date = START_DATE,
                           end_date   = NULL,
                           cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  headline <- c(
    GDP_Nominal = "GDP",
    GDP_Real    = "GDPC1"
  )
  growth <- c(
    GDP_Growth_QoQ = "A191RL1Q225SBEA"
  )
  per_cap <- c(
    GDP_Per_Capita_Real = "A939RX0Q048SBEA"
  )

  h_df <- download_fred_series(headline, start_date, end_date)
  g_df <- download_fred_series(growth, start_date, end_date)
  p_df <- download_fred_series(per_cap, start_date, end_date)

  combined <- h_df %>%
    full_join(g_df, by = "date") %>%
    full_join(p_df, by = "date") %>%
    arrange(date)

  list(headline = h_df, growth = g_df, per_capita = p_df, combined = combined)
}


#' Load GDP expenditure components (C + I + G + NX) from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `consumption`, `investment`, `government`,
#'         `trade`, `combined`.
load_gdp_components_data <- function(start_date = START_DATE,
                                      end_date   = NULL,
                                      cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  consumption <- c(
    PCE_Total        = "PCE",
    PCE_Goods        = "DGDSRC1",
    PCE_Durable      = "PCDG",
    PCE_Nondurable   = "PCND",
    PCE_Services     = "PCESV"
  )
  investment <- c(
    Investment_Total         = "GPDI",
    Investment_Fixed         = "FPI",
    Investment_Nonresidential = "PNFI",
    Investment_Residential    = "PRFI"
  )
  government <- c(
    Govt_Total      = "GCE",
    Govt_Federal    = "FGCE",
    Govt_State_Local = "SLCE"
  )
  trade <- c(
    Exports_Total = "EXPGS",
    Imports_Total = "IMPGS",
    Net_Exports   = "NETEXP"
  )

  c_df <- download_fred_series(consumption, start_date, end_date)
  i_df <- download_fred_series(investment, start_date, end_date)
  g_df <- download_fred_series(government, start_date, end_date)
  t_df <- download_fred_series(trade, start_date, end_date)

  combined <- c_df %>%
    full_join(i_df, by = "date") %>%
    full_join(g_df, by = "date") %>%
    full_join(t_df, by = "date") %>%
    arrange(date)

  list(consumption = c_df, investment = i_df,
       government = g_df, trade = t_df, combined = combined)
}


#' Load GDP by industry / sector from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `industries`, `combined`.
load_gdp_by_industry_data <- function(start_date = START_DATE,
                                       end_date   = NULL,
                                       cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  ind <- c(
    VA_Private     = "VAPGDP",
    VA_Manufacturing = "VAGDPMF",
    VA_Finance     = "VAGDPFI",
    VA_Information = "VAGDPIF",
    VA_Government  = "VAGDPGV"
  )
  ind_df <- download_fred_series(ind, start_date, end_date)
  list(industries = ind_df, combined = ind_df)
}


#' Load comprehensive GDP data.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `headline`, `components`, `industry`, `combined`.
load_comprehensive_gdp_data <- function(start_date = START_DATE,
                                         end_date   = NULL,
                                         cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  gdp_h <- load_gdp_data(start_date, end_date, cache_path)
  gdp_c <- load_gdp_components_data(start_date, end_date, cache_path)
  gdp_i <- load_gdp_by_industry_data(start_date, end_date, cache_path)

  combined <- gdp_h$combined %>%
    full_join(gdp_c$combined, by = "date") %>%
    full_join(gdp_i$combined, by = "date") %>%
    arrange(date)

  list(headline = gdp_h, components = gdp_c, industry = gdp_i,
       combined = combined)
}


# =============================================================================
# FRED MACRO LOADERS — Employment
# =============================================================================

#' Load employment data from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `payrolls`, `unemployment`, `labor_force`, `combined`.
load_employment_data <- function(start_date = START_DATE,
                                  end_date   = NULL,
                                  cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  payrolls <- c(
    Nonfarm_Payrolls = "PAYEMS",
    Private_Payrolls = "USPRIV"
  )
  unemployment <- c(
    Unemployment_Rate = "UNRATE",
    U6_Rate           = "U6RATE"
  )
  labor_force <- c(
    Labor_Force_Participation = "CIVPART",
    Employment_Pop_Ratio      = "EMRATIO"
  )

  pay_df <- download_fred_series(payrolls, start_date, end_date)
  unemp_df <- download_fred_series(unemployment, start_date, end_date)
  lf_df <- download_fred_series(labor_force, start_date, end_date)

  combined <- pay_df %>%
    full_join(unemp_df, by = "date") %>%
    full_join(lf_df, by = "date") %>%
    arrange(date)

  list(payrolls = pay_df, unemployment = unemp_df,
       labor_force = lf_df, combined = combined)
}


#' Load jobless claims data from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `claims`, `combined`.
load_jobless_claims_data <- function(start_date = START_DATE,
                                      end_date   = NULL,
                                      cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  claims <- c(
    Initial_Claims    = "ICSA",
    Continuing_Claims = "CCSA",
    Initial_4WMA      = "IC4WSA",
    Insured_Unemp_Rate = "IURSA"
  )
  claims_df <- download_fred_series(claims, start_date, end_date)
  list(claims = claims_df, combined = claims_df)
}


#' Load wages and hours worked data from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `wages`, `hours`, `combined`.
load_wages_hours_data <- function(start_date = START_DATE,
                                   end_date   = NULL,
                                   cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  wages <- c(
    Avg_Hourly_Earnings = "CES0500000003",
    Employment_Cost_Idx = "ECIWAG",
    Unit_Labor_Costs    = "ULCNFB"
  )
  hours <- c(
    Avg_Weekly_Hours = "AWHAETP"
  )

  w_df <- download_fred_series(wages, start_date, end_date)
  h_df <- download_fred_series(hours, start_date, end_date)

  combined <- full_join(w_df, h_df, by = "date") %>% arrange(date)
  list(wages = w_df, hours = h_df, combined = combined)
}


#' Load JOLTS data from FRED.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `openings`, `turnover`, `combined`.
load_jolts_data <- function(start_date = START_DATE,
                             end_date   = NULL,
                             cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  openings <- c(
    Job_Openings      = "JTSJOL",
    Job_Openings_Rate = "JTSJOR"
  )
  turnover <- c(
    Hires      = "JTSHIL",
    Hires_Rate = "JTSHIR",
    Quits      = "JTSQUL",
    Quits_Rate = "JTSQUR",
    Layoffs    = "JTSLDL"
  )

  o_df <- download_fred_series(openings, start_date, end_date)
  t_df <- download_fred_series(turnover, start_date, end_date)

  combined <- full_join(o_df, t_df, by = "date") %>% arrange(date)
  list(openings = o_df, turnover = t_df, combined = combined)
}


#' Load comprehensive employment data.
#'
#' @inheritParams load_inflation_data
#' @return Named list with `employment`, `claims`, `wages_hours`,
#'         `jolts`, `combined`.
load_comprehensive_employment_data <- function(start_date = START_DATE,
                                                end_date   = NULL,
                                                cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  emp    <- load_employment_data(start_date, end_date, cache_path)
  claims <- load_jobless_claims_data(start_date, end_date, cache_path)
  wh     <- load_wages_hours_data(start_date, end_date, cache_path)
  jolts  <- load_jolts_data(start_date, end_date, cache_path)

  combined <- emp$combined %>%
    full_join(claims$combined, by = "date") %>%
    full_join(wh$combined, by = "date") %>%
    full_join(jolts$combined, by = "date") %>%
    arrange(date)

  list(employment = emp, claims = claims, wages_hours = wh,
       jolts = jolts, combined = combined)
}


# =============================================================================
# FRED MACRO LOADERS — Additional Macro
# =============================================================================

#' Load additional macro indicators (Sentiment, Housing, Dollar Index).
#'
#' @inheritParams load_inflation_data
#' @return A tibble with date and macro series columns.
load_additional_macro_data <- function(start_date = START_DATE,
                                        end_date   = NULL,
                                        cache_path = CACHE_PATH) {
  if (is.null(end_date)) end_date <- as.character(Sys.Date())

  series <- c(
    Consumer_Sentiment = "UMCSENT",
    Housing_Starts     = "HOUST",
    Home_Price_Index   = "CSUSHPINSA",
    Dollar_Index       = "DTWEXBGS"
  )
  download_fred_series(series, start_date, end_date)
}


# =============================================================================
# ORCHESTRATION — Culture War Companies & Control Firms
# =============================================================================

#' Extract unique company tickers from the culture war dataset.
#'
#' @param culture_war_data Tibble of culture war events.
#' @param include_controls Logical; if TRUE, also include control tickers.
#' @return Character vector of unique ticker symbols.
load_culture_war_companies <- function(culture_war_data,
                                        include_controls = TRUE) {
  ticker_cols  <- c("Ticker", "ticker", "TICKER", "Symbol")
  control_cols <- c("Control Ticker", "Control_Ticker", "CONTROL_TICKER")

  ticker_col <- NULL
  for (col in ticker_cols) {
    if (col %in% names(culture_war_data)) { ticker_col <- col; break }
  }
  if (is.null(ticker_col)) {
    stop("Cannot find ticker column. Available: ",
         paste(names(culture_war_data), collapse = ", "))
  }

  tickers <- culture_war_data[[ticker_col]] %>%
    unique() %>%
    na.omit() %>%
    as.character()
  tickers <- tickers[!tickers %in% c("", "Private", "N/A", "NA", "None")]

  if (include_controls) {
    for (col in control_cols) {
      if (col %in% names(culture_war_data)) {
        controls <- culture_war_data[[col]] %>%
          unique() %>%
          na.omit() %>%
          as.character()
        controls <- controls[!controls %in% c("", "Private", "N/A", "NA", "None")]
        controls <- setdiff(controls, tickers)
        tickers <- c(tickers, controls)
        message("Including ", length(controls), " control tickers")
        break
      }
    }
  }
  tickers
}


#' Build a control companies table from culture war data.
#'
#' @param culture_war_data Tibble of culture war events.
#' @return A tibble with CONTROL_TICKER, CONTROL_FIRM, TREATMENT_TICKER, etc.
build_control_companies <- function(culture_war_data) {
  # Detect column names
  find_col <- function(candidates) {
    for (c in candidates) {
      if (c %in% names(culture_war_data)) return(c)
    }
    NA_character_
  }

  ct_col   <- find_col(c("Control Ticker", "Control_Ticker", "CONTROL_TICKER"))
  cf_col   <- find_col(c("Control Firm", "Control_Firm", "CONTROL_FIRM"))
  tk_col   <- find_col(c("Ticker", "ticker", "TICKER"))
  co_col   <- find_col(c("Company", "company", "COMPANY"))
  ind_col  <- find_col(c("Industry", "industry", "INDUSTRY"))
  naics_col <- find_col(c("NAICS Code", "NAICS_Code", "NAICS_CODE"))

  if (is.na(ct_col)) {
    warning("No control ticker column found")
    return(tibble())
  }

  df <- culture_war_data %>%
    filter(!is.na(.data[[ct_col]]), trimws(.data[[ct_col]]) != "")

  safe_get <- function(row, col) {
    if (is.na(col)) return("")
    as.character(row[[col]])
  }

  result <- df %>%
    transmute(
      CONTROL_TICKER    = trimws(.data[[ct_col]]),
      CONTROL_FIRM      = if (!is.na(cf_col)) .data[[cf_col]] else "",
      TREATMENT_TICKER  = if (!is.na(tk_col)) .data[[tk_col]] else "",
      TREATMENT_COMPANY = if (!is.na(co_col)) .data[[co_col]] else "",
      INDUSTRY          = if (!is.na(ind_col)) .data[[ind_col]] else "",
      NAICS_CODE        = if (!is.na(naics_col)) .data[[naics_col]] else ""
    )

  message("Built CONTROL_COMPANIES table: ", nrow(result), " rows (",
          n_distinct(result$CONTROL_TICKER), " unique controls, ",
          n_distinct(result$TREATMENT_TICKER), " unique treatments)")
  result
}


# =============================================================================
# DATA CLEANING
# =============================================================================

#' Clean a single data frame (forward fill, interpolation, drop all-NA rows).
#'
#' @param df     A tibble or data.frame.
#' @param method Cleaning method: "ffill", "interpolate", or "drop".
#' @param max_gap Maximum consecutive NAs to fill.
#' @return Cleaned tibble.
clean_dataframe <- function(df, method = "ffill", max_gap = 5) {
  if (is.null(df) || nrow(df) == 0) return(df)

  # Ensure sorted by date if a date column exists
  if ("date" %in% names(df)) {
    df <- df %>% arrange(date)
  }

  numeric_cols <- names(df)[sapply(df, is.numeric)]

  if (method == "ffill") {
    df <- df %>%
      mutate(across(all_of(numeric_cols), ~ zoo::na.locf(., na.rm = FALSE, maxgap = max_gap)))
  } else if (method == "interpolate") {
    df <- df %>%
      mutate(across(all_of(numeric_cols), ~ zoo::na.approx(., na.rm = FALSE, maxgap = max_gap)))
  } else if (method == "drop") {
    df <- df %>% drop_na()
  }

  # Drop rows where all numeric columns are NA
  df <- df %>%
    filter(rowSums(!is.na(pick(all_of(numeric_cols)))) > 0)

  df
}


#' Clean all datasets in a data dictionary (list of datasets).
#'
#' Applies forward fill to time-series data and keeps cross-sectional
#' data as-is.
#'
#' @param data_dict Named list of datasets.
#' @param verbose   Logical; print progress.
#' @return Named list of cleaned datasets.
clean_all_data <- function(data_dict, verbose = TRUE) {
  if (verbose) message("=== Cleaning All Datasets ===")

  time_series_keys <- c(
    "inflationdata", "inflation_expectations", "comprehensive_inflation",
    "treasury_yields", "policy_rates", "credit_spreads", "comprehensive_rates",
    "industrial_production", "ip_growth", "comprehensive_ip",
    "money_supply", "money_velocity", "fed_balance_sheet", "comprehensive_m2",
    "gdp_data", "gdp_components", "gdp_industry", "comprehensive_gdp",
    "employment_data", "jobless_claims", "wages_hours", "jolts_data",
    "comprehensive_employment", "additional_macro", "vixdata"
  )

  cleaned <- list()
  for (key in names(data_dict)) {
    data <- data_dict[[key]]
    if (is.null(data)) {
      cleaned[[key]] <- NULL
      if (verbose) message("  ", key, ": Skipped (NULL)")
      next
    }

    tryCatch({
      if (key == "stockdata" && is.list(data) && !is.data.frame(data)) {
        cleaned[[key]] <- lapply(data, function(d) {
          if (!is.null(d) && nrow(d) > 0) clean_dataframe(d) else d
        })
        if (verbose) message("  ", key, ": Cleaned ", length(cleaned[[key]]), " tickers")

      } else if (key %in% time_series_keys) {
        if (is.list(data) && !is.data.frame(data)) {
          cleaned[[key]] <- lapply(data, function(d) {
            if (is.data.frame(d) && nrow(d) > 0) clean_dataframe(d) else d
          })
          if (verbose) message("  ", key, ": Cleaned sub-frames")
        } else if (is.data.frame(data)) {
          cleaned[[key]] <- clean_dataframe(data)
          if (verbose) message("  ", key, ": Cleaned (", nrow(data), " rows)")
        } else {
          cleaned[[key]] <- data
        }

      } else {
        # Cross-sectional data — keep as is
        cleaned[[key]] <- data
        if (verbose) {
          if (is.data.frame(data)) {
            message("  ", key, ": Kept as-is (", nrow(data), " rows)")
          } else {
            message("  ", key, ": Kept as-is")
          }
        }
      }
    }, error = function(e) {
      warning(key, ": error during cleaning — ", conditionMessage(e))
      cleaned[[key]] <<- data
    })
  }

  if (verbose) message("=== Data cleaning complete ===")
  cleaned
}


# =============================================================================
# POLITICAL EVENTS (simplified)
# =============================================================================

#' Load political events data (congressional votes, EOs, court decisions).
#'
#' This is a simplified version that loads from a local CSV cache or
#' constructs a skeleton DataFrame. The full Python version scrapes
#' GovTrack, Federal Register, and SCDB; this R version provides the
#' interface and loads cached data if available.
#'
#' @param start_date Start date.
#' @param end_date   End date.
#' @param cache_dir  Directory for cached event data.
#' @return A tibble of political events.
load_political_events <- function(start_date = START_DATE,
                                   end_date   = END_DATE,
                                   cache_dir  = "./data/political_events") {
  cache_file <- file.path(cache_dir, "political_events.csv")
  if (file.exists(cache_file)) {
    message("Loading cached political events from ", cache_file)
    df <- read_csv(cache_file, show_col_types = FALSE)
    if ("date" %in% names(df)) {
      df <- df %>%
        mutate(date = as.Date(date)) %>%
        filter(date >= as.Date(start_date), date <= as.Date(end_date))
    }
    return(df)
  }
  message("Political events cache not found at ", cache_file)
  message("  Run the Python pipeline first to generate this data.")
  tibble(
    date        = as.Date(character(0)),
    event_type  = character(0),
    description = character(0),
    policy_area = character(0),
    source      = character(0)
  )
}


# =============================================================================
# POLITICAL EXPOSURE (simplified)
# =============================================================================

#' Load political exposure data (lobbying, PAC contributions).
#'
#' @param tickers   Character vector of tickers (optional).
#' @param start_date Start date.
#' @param end_date   End date.
#' @param cache_dir  Directory for cached exposure data.
#' @return A tibble of firm-level political exposure.
load_political_exposure <- function(tickers   = NULL,
                                     start_date = START_DATE,
                                     end_date   = END_DATE,
                                     cache_dir  = "./data/political_exposure") {
  cache_file <- file.path(cache_dir, "political_exposure.csv")
  if (file.exists(cache_file)) {
    message("Loading cached political exposure from ", cache_file)
    df <- read_csv(cache_file, show_col_types = FALSE)
    if (!is.null(tickers) && "ticker" %in% names(df)) {
      df <- df %>% filter(ticker %in% tickers)
    }
    return(df)
  }
  message("Political exposure cache not found at ", cache_file)
  tibble(
    ticker         = character(0),
    year           = integer(0),
    lobbying_total = double(0),
    pac_total      = double(0)
  )
}


# =============================================================================
# ORTHOGONALIZATION — Baker-Wurgler (2006) Purging
# =============================================================================

# Party control intervals (start, end, is_democrat)
.PRESIDENT_PARTY <- tribble(
  ~start,        ~end,          ~dem,
  "1993-01-20",  "2001-01-19",  1L,
  "2001-01-20",  "2009-01-19",  0L,
  "2009-01-20",  "2017-01-19",  1L,
  "2017-01-20",  "2021-01-19",  0L,
  "2021-01-20",  "2025-01-19",  1L,
  "2025-01-20",  "2029-01-19",  0L
)

.SENATE_MAJORITY <- tribble(
  ~start,        ~end,          ~dem,
  "2000-01-01",  "2001-01-02",  0L,
  "2001-01-03",  "2001-06-05",  0L,
  "2001-06-06",  "2003-01-02",  1L,
  "2003-01-03",  "2007-01-03",  0L,
  "2007-01-04",  "2011-01-04",  1L,
  "2011-01-05",  "2015-01-05",  1L,
  "2015-01-06",  "2021-01-19",  0L,
  "2021-01-20",  "2023-01-02",  1L,
  "2023-01-03",  "2025-01-02",  1L,
  "2025-01-03",  "2027-01-02",  0L
)

.HOUSE_MAJORITY <- tribble(
  ~start,        ~end,          ~dem,
  "2000-01-01",  "2007-01-03",  0L,
  "2007-01-04",  "2011-01-04",  1L,
  "2011-01-05",  "2019-01-02",  0L,
  "2019-01-03",  "2023-01-02",  1L,
  "2023-01-03",  "2025-01-02",  0L,
  "2025-01-03",  "2027-01-02",  0L
)


#' Build a party-control time series from interval definitions.
#'
#' @param intervals Tibble with start, end, dem columns.
#' @param freq      "month" or "day".
#' @param start     Series start date.
#' @param end       Series end date.
#' @return A tibble with `date` and `value` columns.
.build_party_series <- function(intervals, freq = "month",
                                start = "2000-01-01", end = "2025-12-31") {
  dates <- seq.Date(as.Date(start), as.Date(end), by = freq)
  vals  <- rep(NA_real_, length(dates))

  for (i in seq_len(nrow(intervals))) {
    s <- as.Date(intervals$start[i])
    e <- as.Date(intervals$end[i])
    mask <- dates >= s & dates <= e
    vals[mask] <- intervals$dem[i]
  }
  # Forward-fill then back-fill
  vals <- zoo::na.locf(vals, na.rm = FALSE)
  vals <- zoo::na.locf(vals, na.rm = FALSE, fromLast = TRUE)
  tibble(date = dates, value = vals)
}


#' Build a DataFrame of political proxy variables.
#'
#' @param freq  "month", "quarter", or "day".
#' @param start Start date.
#' @param end   End date.
#' @return A tibble with PRES_DEM, SENATE_DEM, HOUSE_DEM, UNIFIED_GOV,
#'         and optionally EPU columns.
build_political_proxies <- function(freq  = "month",
                                     start = "2000-01-01",
                                     end   = "2025-12-31") {
  pres   <- .build_party_series(.PRESIDENT_PARTY, freq, start, end)
  senate <- .build_party_series(.SENATE_MAJORITY, freq, start, end)
  house  <- .build_party_series(.HOUSE_MAJORITY, freq, start, end)

  df <- tibble(
    date       = pres$date,
    PRES_DEM   = pres$value,
    SENATE_DEM = senate$value,
    HOUSE_DEM  = house$value
  ) %>%
    mutate(UNIFIED_GOV = as.numeric(
      PRES_DEM == SENATE_DEM & PRES_DEM == HOUSE_DEM
    ))

  # Try to add EPU from FRED
  tryCatch({
    validate_fred_api_key()
    epu_raw <- fredr(
      series_id        = "USEPUINDXM",
      observation_start = as.Date(start),
      observation_end   = as.Date(end)
    )
    if (nrow(epu_raw) > 0) {
      epu <- tibble(
        date = floor_date(epu_raw$date, "month"),
        EPU  = epu_raw$value
      )
      df <- left_join(df, epu, by = "date") %>%
        mutate(EPU = zoo::na.approx(EPU, na.rm = FALSE))
    }
  }, error = function(e) {
    message("EPU index not available: ", conditionMessage(e))
  })

  message("Political proxies: ", nrow(df), " rows, ", ncol(df), " columns")
  df
}


#' Construct a time-series culture index from culture war events.
#'
#' @param events_df Culture war events tibble.
#' @param freq      "month" or "quarter".
#' @param method    "net_score", "count_weighted", or "intensity".
#' @return A tibble with `date` and `CULTURE_RAW` columns.
build_culture_index <- function(events_df,
                                 freq   = "month",
                                 method = "net_score") {
  df <- events_df

  # Find columns
  date_col <- NULL
  lean_col <- NULL
  for (c in names(df)) {
    if (grepl("event", tolower(c)) && grepl("date", tolower(c))) date_col <- c
    if (grepl("estimated", tolower(c)) && grepl("political", tolower(c))) lean_col <- c
  }
  if (is.null(lean_col)) {
    for (c in names(df)) {
      cl <- tolower(c)
      if (grepl("political", cl) && grepl("lean", cl) && !grepl("justif", cl)) {
        lean_col <- c; break
      }
    }
  }
  if (is.null(date_col) || is.null(lean_col)) {
    stop("Cannot find Event Date or Political Leaning columns. Available: ",
         paste(names(df), collapse = ", "))
  }

  df <- df %>%
    mutate(.date = as.Date(.data[[date_col]])) %>%
    filter(!is.na(.date))

  lean_map <- c("liberal" = 1, "conservative" = -1, "mixed" = 0)
  df <- df %>%
    mutate(.score = lean_map[tolower(trimws(.data[[lean_col]]))])
  df$.score[is.na(df$.score)] <- 0

  df <- df %>%
    mutate(.period = floor_date(.date, freq))

  if (method == "net_score") {
    culture <- df %>% group_by(.period) %>% summarise(CULTURE_RAW = sum(.score), .groups = "drop")
  } else if (method == "count_weighted") {
    culture <- df %>% group_by(.period) %>%
      summarise(CULTURE_RAW = n() * mean(.score), .groups = "drop")
  } else if (method == "intensity") {
    culture <- df %>% group_by(.period) %>%
      summarise(CULTURE_RAW = n(), .groups = "drop")
  } else {
    stop("Unknown method: ", method)
  }

  # Fill gaps
  full_dates <- seq.Date(min(culture$.period), max(culture$.period), by = freq)
  culture <- tibble(date = full_dates) %>%
    left_join(culture, by = c("date" = ".period")) %>%
    replace_na(list(CULTURE_RAW = 0))

  message("Culture index (", method, "): ", nrow(culture), " periods, ",
          "mean=", round(mean(culture$CULTURE_RAW), 2),
          ", std=", round(sd(culture$CULTURE_RAW), 2))
  culture
}


#' Orthogonalize culture factor against political proxies.
#'
#' Implements the Baker-Wurgler (2006) purging approach:
#'   Culture_t = alpha + beta' * Political_t + epsilon_t
#'
#' @param culture   Tibble with `date` and `CULTURE_RAW` columns.
#' @param political Tibble from build_political_proxies().
#' @param method    "full_sample", "rolling", or "expanding".
#' @param window    Rolling window size in periods.
#' @param min_obs   Minimum observations.
#' @return A named list with `culture_raw`, `culture_orthogonal`,
#'         `first_stage_r2`, `coefficients`, etc.
orthogonalize_culture <- function(culture,
                                   political,
                                   method  = "full_sample",
                                   window  = 60,
                                   min_obs = 24) {
  # Merge on date
  merged <- inner_join(culture, political, by = "date") %>% drop_na()

  if (nrow(merged) < min_obs) {
    stop("Insufficient observations after alignment: ",
         nrow(merged), " < ", min_obs)
  }

  y <- merged$CULTURE_RAW
  proxy_cols <- setdiff(names(political), "date")
  proxy_cols <- proxy_cols[proxy_cols %in% names(merged)]
  X <- as.matrix(merged[, proxy_cols])
  X <- cbind(const = 1, X)

  message("Orthogonalizing culture: ", nrow(merged), " obs, ",
          length(proxy_cols), " proxies (", paste(proxy_cols, collapse = ", "),
          "), method=", method)

  if (method == "full_sample") {
    fit <- lm(y ~ X - 1)
    residuals <- fit$residuals
    r2 <- summary(fit)$r.squared
    adj_r2 <- summary(fit)$adj.r.squared

    coef_summary <- summary(fit)$coefficients
    coef_df <- tibble(
      VARIABLE    = rownames(coef_summary),
      COEFFICIENT = coef_summary[, 1],
      STD_ERROR   = coef_summary[, 2],
      T_STAT      = coef_summary[, 3],
      P_VALUE     = coef_summary[, 4]
    )
    # Clean variable names
    coef_df$VARIABLE <- gsub("^X", "", coef_df$VARIABLE)

    message("  Full-sample R2=", round(r2, 4),
            ", Adj-R2=", round(adj_r2, 4))

  } else if (method %in% c("rolling", "expanding")) {
    n <- length(y)
    residuals <- rep(NA_real_, n)
    start_idx <- if (method == "rolling") window else min_obs

    for (i in seq(start_idx + 1, n)) {
      idx_start <- if (method == "rolling") max(1, i - window) else 1
      y_win <- y[idx_start:(i - 1)]
      X_win <- X[idx_start:(i - 1), , drop = FALSE]

      tryCatch({
        fit_win <- lm.fit(X_win, y_win)
        y_pred <- sum(X[i, ] * fit_win$coefficients)
        residuals[i] <- y[i] - y_pred
      }, error = function(e) {})
    }

    # Full-sample fit for reporting
    fit <- lm(y ~ X - 1)
    r2 <- summary(fit)$r.squared
    adj_r2 <- summary(fit)$adj.r.squared
    coef_summary <- summary(fit)$coefficients
    coef_df <- tibble(
      VARIABLE    = gsub("^X", "", rownames(coef_summary)),
      COEFFICIENT = coef_summary[, 1],
      STD_ERROR   = coef_summary[, 2],
      T_STAT      = coef_summary[, 3],
      P_VALUE     = coef_summary[, 4]
    )
    valid_count <- sum(!is.na(residuals))
    message("  ", method, " (w=", window, "): ", valid_count, "/", n,
            " valid residuals, full-sample R2=", round(r2, 4))
  } else {
    stop("Unknown method: ", method)
  }

  ortho_df <- tibble(date = merged$date, CULTURE_ORTHOGONAL = residuals)

  list(
    culture_raw        = culture,
    culture_orthogonal = ortho_df,
    first_stage_r2     = r2,
    first_stage_adj_r2 = adj_r2,
    coefficients       = coef_df,
    proxies_used       = proxy_cols,
    method             = method,
    window_size        = if (method == "rolling") window else NULL,
    n_obs              = nrow(merged)
  )
}


#' Run the complete orthogonalization pipeline.
#'
#' @param events_df      Culture war events tibble.
#' @param freq           Time series frequency ("month").
#' @param culture_method Culture index method.
#' @param ortho_method   Orthogonalization method.
#' @param rolling_window Window for rolling estimation.
#' @param output_dir     Output directory for CSVs.
#' @return Orthogonalization result list.
run_orthogonalization_pipeline <- function(events_df,
                                            freq           = "month",
                                            culture_method = "net_score",
                                            ortho_method   = "full_sample",
                                            rolling_window = 60,
                                            output_dir     = "./output") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  message("Step 1: Building political proxy variables...")
  political <- build_political_proxies(freq = freq)

  message("Step 2: Constructing raw culture index...")
  culture <- build_culture_index(events_df, freq = freq, method = culture_method)

  message("Step 3: Orthogonalizing culture factor...")
  result <- orthogonalize_culture(
    culture, political, method = ortho_method, window = rolling_window
  )

  # Save results
  culture_out <- culture %>%
    left_join(result$culture_orthogonal, by = "date")
  write_csv(culture_out, file.path(output_dir, "culture_factors.csv"))
  write_csv(result$coefficients, file.path(output_dir, "first_stage_coefficients.csv"))

  message("=" %>% strrep(60))
  message("ORTHOGONALIZATION RESULTS")
  message("  First-stage R2 = ", round(result$first_stage_r2, 4),
          " (", round(result$first_stage_r2 * 100, 1),
          "% of culture variance is political)")
  message("  Method: ", result$method)
  message("  N observations: ", result$n_obs)

  result
}

message("clean.R loaded successfully. Functions available:")
message("  Data: import_culture_war_data, load_culture_war_companies, build_control_companies")
message("  Market: get_stock_data, download_vix_data, download_fama_french_factors")
message("  FRED: load_inflation_data, load_treasury_yields, load_employment_data, ...")
message("  Clean: clean_dataframe, clean_all_data")
message("  Political: build_political_proxies, build_culture_index, orthogonalize_culture")
