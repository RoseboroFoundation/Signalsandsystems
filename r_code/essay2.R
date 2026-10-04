# =============================================================================
# Essay 2 -- NLP Pipeline for Culture War Event Analysis
#
# Scores news articles and SEC filings for sentiment and political alignment.
# Uses tidytext with Loughran-McDonald financial lexicon (R equivalent of
# FinBERT from the Python pipeline).
#
# References:
#   Loughran, T. & McDonald, B. (2011). J. Finance, 66(1).
#   Gentzkow, M. & Shapiro, J. (2010). Econometrica, 78(1).
#
# Author: Ashley Roseboro
# =============================================================================

library(tidyverse)
library(tidytext)
library(sentimentr)
library(broom)

# -- Configuration -----------------------------------------------------------
NEWS_WINDOW_DAYS   <- 30L    # calendar days around event for news
FILING_WINDOW_DAYS <- 180L   # calendar days before event for filings
N_DISTINCTIVE      <- 200L   # distinctive phrases per party
FILING_WEIGHT      <- 3.0
NEWS_WEIGHT        <- 1.0
W_DISTINCTIVE      <- 0.4
W_COSINE           <- 0.6    # R version uses 2 signals (no FinBERT stance)
MIN_AGREEMENT_RATE <- 0.55

# =============================================================================
# Score News Sentiment
# =============================================================================
#' Score news articles using sentimentr (sentence-level) and
#' Loughran-McDonald financial lexicon via tidytext.
#'
#' @param news_df data.frame with TITLE, SNIPPET (or TEXT), TICKER, DATE
#' @return data.frame with sentiment columns added:
#'   SENTIMENTR_SCORE, LM_POSITIVE, LM_NEGATIVE, LM_NET, SENTIMENT, SENT_WEIGHTED
score_news_sentiment <- function(news_df) {
  df <- news_df

  # Build text field
  if (!"TEXT" %in% names(df)) {
    title   <- if ("TITLE" %in% names(df)) replace_na(df$TITLE, "") else ""
    snippet <- if ("SNIPPET" %in% names(df)) replace_na(df$SNIPPET, "") else ""
    df$TEXT <- trimws(paste(title, snippet))
  }

  df <- df %>% filter(nchar(TEXT) > 10)
  if (nrow(df) == 0) {
    warning("No news articles with sufficient text")
    return(df)
  }

  message("Scoring ", nrow(df), " news articles...")

  # sentimentr: sentence-level sentiment
  sent_scores <- sentiment_by(df$TEXT, by = seq_len(nrow(df)))
  df$SENTIMENTR_SCORE <- sent_scores$ave_sentiment

  # Loughran-McDonald via tidytext
  lm_lexicon <- get_sentiments("loughran")

  # Tokenize and score
  tokens <- df %>%
    mutate(.ROW_ID = row_number()) %>%
    unnest_tokens(word, TEXT) %>%
    inner_join(lm_lexicon, by = "word")

  lm_summary <- tokens %>%
    group_by(.ROW_ID) %>%
    summarise(
      LM_POSITIVE = sum(sentiment == "positive"),
      LM_NEGATIVE = sum(sentiment == "negative"),
      LM_NET      = LM_POSITIVE - LM_NEGATIVE,
      N_LM_WORDS  = n(),
      .groups     = "drop"
    )

  df <- df %>%
    mutate(.ROW_ID = row_number()) %>%
    left_join(lm_summary, by = ".ROW_ID") %>%
    mutate(
      LM_POSITIVE = replace_na(LM_POSITIVE, 0L),
      LM_NEGATIVE = replace_na(LM_NEGATIVE, 0L),
      LM_NET      = replace_na(LM_NET, 0L),
      N_LM_WORDS  = replace_na(N_LM_WORDS, 0L)
    ) %>%
    select(-.ROW_ID)

  # Composite sentiment: combine sentimentr and LM net
  df$SENTIMENT <- df$SENTIMENTR_SCORE
  df$SENT_WEIGHTED <- df$SENTIMENT  # equal weighting; extend if needed

  message("News sentiment scored: ",
          sum(df$SENTIMENT > 0, na.rm = TRUE), " positive, ",
          sum(df$SENTIMENT < 0, na.rm = TRUE), " negative, ",
          sum(df$SENTIMENT == 0, na.rm = TRUE), " neutral")

  return(df)
}

# =============================================================================
# Score Filing Sentiment
# =============================================================================
#' Score SEC filing sections using sentimentr and Loughran-McDonald.
#'
#' @param filings_df data.frame with TICKER, FORM_TYPE, FILING_DATE, SECTION, TEXT
#' @return data.frame with sentiment columns per section
score_filing_sentiment <- function(filings_df) {
  if (nrow(filings_df) == 0) return(data.frame())

  message("Scoring ", nrow(filings_df), " filing sections...")

  results <- filings_df %>%
    mutate(
      .ROW_ID       = row_number(),
      FILING_DATE   = as.Date(FILING_DATE),
      TEXT_LENGTH   = nchar(TEXT)
    )

  # sentimentr
  sent_scores <- sentiment_by(results$TEXT, by = results$.ROW_ID)

  # LM lexicon
  lm_lexicon <- get_sentiments("loughran")

  lm_scores <- results %>%
    select(.ROW_ID, TEXT) %>%
    unnest_tokens(word, TEXT) %>%
    inner_join(lm_lexicon, by = "word") %>%
    group_by(.ROW_ID) %>%
    summarise(
      PCT_POSITIVE = mean(sentiment == "positive"),
      PCT_NEGATIVE = mean(sentiment == "negative"),
      PCT_NEUTRAL  = mean(sentiment %in% c("uncertainty", "litigious",
                                             "constraining", "superfluous")),
      N_LM_WORDS   = n(),
      .groups      = "drop"
    )

  results <- results %>%
    left_join(
      sent_scores %>% select(.ROW_ID = element_id, SENT_MEAN = ave_sentiment),
      by = ".ROW_ID"
    ) %>%
    left_join(lm_scores, by = ".ROW_ID") %>%
    mutate(across(c(PCT_POSITIVE, PCT_NEGATIVE, PCT_NEUTRAL, N_LM_WORDS),
                  ~replace_na(., 0))) %>%
    select(TICKER, FORM_TYPE, FILING_DATE, SECTION, SENT_MEAN,
           PCT_POSITIVE, PCT_NEGATIVE, PCT_NEUTRAL, TEXT_LENGTH)

  message("Filing sentiment: ", nrow(results), " sections scored across ",
          n_distinct(results$TICKER), " tickers")

  return(results)
}

# =============================================================================
# Build Event NLP Panel
# =============================================================================
#' Merge NLP features to event panel for DiD.
#'
#' @param news_scored data.frame from score_news_sentiment()
#' @param filing_scored data.frame from score_filing_sentiment()
#' @param events data.frame with TICKER, EVENT_DATE (or DATE)
#' @param news_window_days integer, days around event for news
#' @param filing_window_days integer, days before event for filings
#' @return data.frame with columns: TICKER, EVENT_DATE, NEWS_SENT_PRE,
#'   NEWS_SENT_POST, NEWS_N_PRE, NEWS_N_POST, FILING_MDA_TONE, FILING_RISK_TONE
build_event_nlp_panel <- function(news_scored = NULL, filing_scored = NULL,
                                   events, news_window_days = NEWS_WINDOW_DAYS,
                                   filing_window_days = FILING_WINDOW_DAYS) {
  if (nrow(events) == 0) {
    warning("No events for NLP panel")
    return(data.frame())
  }

  events <- events %>% mutate(across(any_of(c("EVENT_DATE", "DATE")), as.Date))
  date_col <- if ("EVENT_DATE" %in% names(events)) "EVENT_DATE" else "DATE"

  rows <- list()

  for (i in seq_len(nrow(events))) {
    ticker     <- events$TICKER[i]
    event_date <- events[[date_col]][i]
    event_id   <- paste0(ticker, "_", event_date)

    row <- list(
      TICKER     = ticker,
      EVENT_DATE = event_date,
      EVENT_ID   = event_id
    )

    # News sentiment around event
    if (!is.null(news_scored) && nrow(news_scored) > 0) {
      ticker_news <- news_scored %>%
        filter(TICKER == ticker, !is.na(DATE)) %>%
        mutate(DATE = as.Date(DATE))

      if (nrow(ticker_news) > 0) {
        pre_news  <- ticker_news %>%
          filter(DATE >= event_date - news_window_days, DATE < event_date)
        post_news <- ticker_news %>%
          filter(DATE >= event_date, DATE <= event_date + news_window_days)

        sent_col <- if ("SENT_WEIGHTED" %in% names(ticker_news)) "SENT_WEIGHTED" else "SENTIMENT"
        row$NEWS_SENT_PRE  <- if (nrow(pre_news) > 0) mean(pre_news[[sent_col]], na.rm = TRUE) else NA_real_
        row$NEWS_SENT_POST <- if (nrow(post_news) > 0) mean(post_news[[sent_col]], na.rm = TRUE) else NA_real_
        row$NEWS_N_PRE     <- nrow(pre_news)
        row$NEWS_N_POST    <- nrow(post_news)
      }
    }

    # Filing sentiment
    if (!is.null(filing_scored) && nrow(filing_scored) > 0) {
      ticker_filings <- filing_scored %>%
        filter(TICKER == ticker, FILING_DATE < event_date,
               FILING_DATE >= event_date - filing_window_days) %>%
        arrange(desc(FILING_DATE))

      if (nrow(ticker_filings) == 0) {
        # Fallback: most recent filing
        ticker_filings <- filing_scored %>%
          filter(TICKER == ticker, FILING_DATE < event_date) %>%
          arrange(desc(FILING_DATE)) %>%
          head(4)
      }

      if ("SENT_MEAN" %in% names(ticker_filings)) {
        mda <- ticker_filings %>% filter(SECTION %in% c("MDA", "Item 7 - MD&A", "Item 2 - MD&A"))
        if (nrow(mda) > 0) {
          row$FILING_MDA_TONE  <- mda$SENT_MEAN[1]
          row$FILING_FORM_TYPE <- mda$FORM_TYPE[1]
          row$FILING_DATE      <- mda$FILING_DATE[1]
        }

        risk <- ticker_filings %>% filter(SECTION %in% c("RISK_FACTORS", "Item 1A - Risk Factors"))
        if (nrow(risk) > 0) {
          row$FILING_RISK_TONE <- risk$SENT_MEAN[1]
        }
      }
    }

    rows[[i]] <- as_tibble(row)
  }

  panel <- bind_rows(rows)

  # Fill missing columns
  for (col in c("NEWS_SENT_PRE", "NEWS_SENT_POST", "FILING_MDA_TONE", "FILING_RISK_TONE")) {
    if (!col %in% names(panel)) panel[[col]] <- NA_real_
  }
  for (col in c("NEWS_N_PRE", "NEWS_N_POST")) {
    if (!col %in% names(panel)) panel[[col]] <- 0L
    panel[[col]] <- replace_na(panel[[col]], 0L)
  }

  # News sentiment change
  panel$NEWS_SENT_CHANGE <- panel$NEWS_SENT_POST - panel$NEWS_SENT_PRE

  message("Event NLP panel: ", nrow(panel), " events, ",
          sum(!is.na(panel$NEWS_SENT_PRE)), " with news, ",
          sum(!is.na(panel$FILING_MDA_TONE)), " with filings")

  return(panel)
}

# =============================================================================
# Full NLP Pipeline Orchestrator
# =============================================================================
#' @param news_df data.frame of raw news articles
#' @param filings_df data.frame of filing sections with TEXT
#' @param events data.frame of culture war events
#' @return named list (NLPAnalysis equivalent):
#'   - news_sentiment, filing_sentiment, event_nlp_panel
#'   - n_articles_scored, n_filings_scored, n_tickers
run_nlp_analysis <- function(news_df = NULL, filings_df = NULL, events) {
  # Score news
  news_scored <- if (!is.null(news_df) && nrow(news_df) > 0) {
    score_news_sentiment(news_df)
  } else {
    data.frame()
  }

  # Score filings
  filing_scored <- if (!is.null(filings_df) && nrow(filings_df) > 0) {
    score_filing_sentiment(filings_df)
  } else {
    data.frame()
  }

  # Build event panel
  event_panel <- build_event_nlp_panel(
    news_scored  = if (nrow(news_scored) > 0) news_scored else NULL,
    filing_scored = if (nrow(filing_scored) > 0) filing_scored else NULL,
    events       = events
  )

  list(
    news_sentiment    = news_scored,
    filing_sentiment  = filing_scored,
    event_nlp_panel   = event_panel,
    n_articles_scored = nrow(news_scored),
    n_filings_scored  = nrow(filing_scored),
    n_tickers         = if (nrow(event_panel) > 0) n_distinct(event_panel$TICKER) else 0L
  )
}

# =============================================================================
# Political Alignment: Compute via TF-IDF on Party Platforms
# =============================================================================
#' Compute text-derived political alignment scores.
#' R equivalent of the Python 3-signal pipeline (without FinBERT stance,
#' so uses 2 signals: distinctive-phrase similarity + raw cosine).
#'
#' @param company_corpus named list: ticker -> text
#' @param platform_corpus data.frame with YEAR, PARTY, TEXT
#' @param events data.frame with TICKER, EVENT_DATE
#' @param conservative_threshold numeric (default 0.05)
#' @param liberal_threshold numeric (default -0.05)
#' @return named list (PoliticalAlignmentResult equivalent)
compute_political_alignment <- function(company_corpus, platform_corpus,
                                         events = NULL,
                                         conservative_threshold = 0.05,
                                         liberal_threshold = -0.05) {
  if (nrow(platform_corpus) == 0) {
    warning("No platform corpus available")
    return(NULL)
  }

  # Extract distinctive phrases
  phrases_result <- extract_distinctive_phrases(platform_corpus, n_phrases = N_DISTINCTIVE)
  phrases_df     <- phrases_result$phrases_df
  vectorizer     <- phrases_result$vectorizer

  # Score each company
  company_rows <- list()
  for (ticker in names(company_corpus)) {
    text <- company_corpus[[ticker]]
    if (length(strsplit(text, "\\s+")[[1]]) < 50) next

    # TF-IDF cosine similarity to R and D platforms
    # Use all available years, average
    r_text <- paste(platform_corpus %>% filter(PARTY == "Republican") %>% pull(TEXT), collapse = " ")
    d_text <- paste(platform_corpus %>% filter(PARTY == "Democratic") %>% pull(TEXT), collapse = " ")

    # Simple word-overlap cosine (lightweight TF-IDF proxy)
    cosine_result <- .compute_tfidf_cosine(text, r_text, d_text)
    cosine_align  <- cosine_result$sim_r - cosine_result$sim_d

    # Distinctive-phrase similarity
    disc_result <- .compute_distinctive_similarity(text, phrases_df, r_text, d_text)
    disc_align  <- disc_result$distinctive_align

    company_rows[[length(company_rows) + 1]] <- data.frame(
      TICKER            = ticker,
      DISTINCTIVE_ALIGN = disc_align,
      COSINE_ALIGN      = cosine_align,
      SIM_REPUBLICAN    = cosine_result$sim_r,
      SIM_DEMOCRATIC    = cosine_result$sim_d,
      stringsAsFactors   = FALSE
    )
  }

  if (length(company_rows) == 0) {
    warning("No alignment scores computed")
    return(NULL)
  }

  company_df <- bind_rows(company_rows)

  # Normalize signals to [-1, +1] using 5th/95th percentile
  .normalize <- function(x) {
    p05 <- quantile(x, 0.05, na.rm = TRUE)
    p95 <- quantile(x, 0.95, na.rm = TRUE)
    if (p95 == p05) return(rep(0, length(x)))
    pmin(pmax((2 * (x - p05) / (p95 - p05) - 1), -1), 1)
  }

  company_df$DISTINCTIVE_ALIGN_NORM <- .normalize(company_df$DISTINCTIVE_ALIGN)
  company_df$COSINE_ALIGN_NORM      <- .normalize(company_df$COSINE_ALIGN)

  # Composite (2-signal, no stance in R)
  w_d <- W_DISTINCTIVE / (W_DISTINCTIVE + W_COSINE)
  w_c <- W_COSINE / (W_DISTINCTIVE + W_COSINE)
  company_df$ALIGNMENT_SCORE <- w_d * company_df$DISTINCTIVE_ALIGN_NORM +
                                 w_c * company_df$COSINE_ALIGN_NORM

  # Classify
  company_df$COMPUTED_LEANING <- case_when(
    company_df$ALIGNMENT_SCORE > conservative_threshold ~ "Conservative",
    company_df$ALIGNMENT_SCORE < liberal_threshold      ~ "Liberal",
    TRUE                                                 ~ "Mixed"
  )

  # Event-level scores
  event_df <- data.frame()
  if (!is.null(events) && nrow(events) > 0) {
    event_df <- events %>%
      mutate(EVENT_DATE = as.Date(EVENT_DATE)) %>%
      inner_join(company_df %>% select(TICKER, ALIGNMENT_SCORE, DISTINCTIVE_ALIGN, COSINE_ALIGN),
                 by = "TICKER")
  }

  list(
    company_scores         = company_df,
    event_scores           = event_df,
    distinctive_phrases    = phrases_df,
    n_companies            = nrow(company_df),
    conservative_threshold = conservative_threshold,
    liberal_threshold      = liberal_threshold
  )
}

# =============================================================================
# Extract Distinctive Phrases via TF-IDF
# =============================================================================
#' Find terms that discriminate between Republican and Democratic platforms.
#'
#' @param platforms data.frame with PARTY, TEXT columns
#' @param n_phrases integer, top N phrases per party
#' @return named list:
#'   - phrases_df: data.frame with PHRASE, PARTY, TFIDF_DIFF, RANK
#'   - vectorizer: NULL (placeholder; R uses tidytext not sklearn)
extract_distinctive_phrases <- function(platforms, n_phrases = N_DISTINCTIVE) {
  # Concatenate all text per party
  party_texts <- platforms %>%
    group_by(PARTY) %>%
    summarise(TEXT = paste(TEXT, collapse = " "), .groups = "drop")

  # Tokenize and compute TF-IDF per party
  tokens <- party_texts %>%
    unnest_tokens(word, TEXT) %>%
    anti_join(stop_words, by = "word") %>%
    count(PARTY, word) %>%
    bind_tf_idf(word, PARTY, n)

  # Pivot to get R and D TF-IDF side by side
  wide <- tokens %>%
    select(PARTY, word, tf_idf) %>%
    pivot_wider(names_from = PARTY, values_from = tf_idf, values_fill = 0) %>%
    rename(TFIDF_R = Republican, TFIDF_D = Democratic) %>%
    mutate(TFIDF_DIFF = TFIDF_R - TFIDF_D)

  # Top N per party
  r_phrases <- wide %>%
    arrange(desc(TFIDF_DIFF)) %>%
    head(n_phrases) %>%
    mutate(PARTY = "Republican", RANK = row_number())

  d_phrases <- wide %>%
    arrange(TFIDF_DIFF) %>%
    head(n_phrases) %>%
    mutate(PARTY = "Democratic", RANK = row_number())

  phrases_df <- bind_rows(r_phrases, d_phrases) %>%
    rename(PHRASE = word) %>%
    select(PHRASE, PARTY, TFIDF_R, TFIDF_D, TFIDF_DIFF, RANK)

  message("Distinctive phrases: ",
          sum(phrases_df$PARTY == "Republican"), " R-distinctive, ",
          sum(phrases_df$PARTY == "Democratic"), " D-distinctive")

  list(
    phrases_df = phrases_df,
    vectorizer = NULL
  )
}

# =============================================================================
# Internal: TF-IDF Cosine Similarity (simplified)
# =============================================================================
.compute_tfidf_cosine <- function(company_text, r_text, d_text) {
  # Word frequency vectors
  .word_freqs <- function(text) {
    words <- tolower(unlist(strsplit(text, "\\W+")))
    words <- words[nchar(words) > 2]
    table(words)
  }

  comp_freq <- .word_freqs(company_text)
  r_freq    <- .word_freqs(r_text)
  d_freq    <- .word_freqs(d_text)

  # Cosine similarity
  .cosine <- function(a, b) {
    all_words <- union(names(a), names(b))
    va <- as.numeric(a[all_words]); va[is.na(va)] <- 0
    vb <- as.numeric(b[all_words]); vb[is.na(vb)] <- 0
    dot <- sum(va * vb)
    na  <- sqrt(sum(va^2))
    nb  <- sqrt(sum(vb^2))
    if (na == 0 || nb == 0) return(0)
    dot / (na * nb)
  }

  list(
    sim_r = .cosine(comp_freq, r_freq),
    sim_d = .cosine(comp_freq, d_freq)
  )
}

# =============================================================================
# Internal: Distinctive-Phrase Similarity
# =============================================================================
.compute_distinctive_similarity <- function(company_text, phrases_df, r_text, d_text) {
  comp_lower <- tolower(company_text)
  distinctive_terms <- tolower(phrases_df$PHRASE)

  # Count occurrences of each distinctive term
  comp_counts <- sapply(distinctive_terms, function(term) {
    length(gregexpr(term, comp_lower, fixed = TRUE)[[1]])
  })
  comp_counts[comp_counts < 0] <- 0

  r_lower <- tolower(r_text)
  d_lower <- tolower(d_text)
  r_counts <- sapply(distinctive_terms, function(term) {
    length(gregexpr(term, r_lower, fixed = TRUE)[[1]])
  })
  r_counts[r_counts < 0] <- 0
  d_counts <- sapply(distinctive_terms, function(term) {
    length(gregexpr(term, d_lower, fixed = TRUE)[[1]])
  })
  d_counts[d_counts < 0] <- 0

  # Cosine similarity on distinctive term counts
  .cosine_vec <- function(a, b) {
    dot <- sum(a * b)
    na  <- sqrt(sum(a^2))
    nb  <- sqrt(sum(b^2))
    if (na == 0 || nb == 0) return(0)
    dot / (na * nb)
  }

  sim_r <- .cosine_vec(comp_counts, r_counts)
  sim_d <- .cosine_vec(comp_counts, d_counts)

  list(
    sim_r             = sim_r,
    sim_d             = sim_d,
    distinctive_align = sim_r - sim_d
  )
}
