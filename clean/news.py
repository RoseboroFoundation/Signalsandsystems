"""News data: Guardian, NYT, Reddit aggregation for culture war events."""

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import praw
from bs4 import BeautifulSoup

from .config import logger, import_culture_war_data

@dataclass
class NewsArticle:
    """Structure for news articles."""
    ticker: str
    company_name: str
    source: str
    title: str
    url: str
    published_date: Optional[datetime]
    snippet: Optional[str]
    culture_war_event: Optional[str] = None
    event_date: Optional[datetime] = None
    search_query: Optional[str] = None
    author: Optional[str] = None
    section: Optional[str] = None
    word_count: Optional[int] = None
    body_text: Optional[str] = None


class CompanyNewsAggregator:
    """Aggregates news from Guardian, Reddit, and NYT (2000-2025)."""

    def __init__(
        self,
        guardian_api_key: Optional[str] = None,
        nyt_api_key: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: Optional[str] = None
    ):
        """
        Initialize the news aggregator.

        Args:
            guardian_api_key: The Guardian API key
            nyt_api_key: New York Times API key
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit user agent string
        """
        self.guardian_api_key = guardian_api_key or os.getenv('GUARDIAN_API_KEY')
        self.nyt_api_key = nyt_api_key or os.getenv('NYT_API_KEY')

        self.reddit = None
        if all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
                logger.info("Reddit client initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize Reddit client: %s", e)

        self.last_nyt_request = None
        self.nyt_requests_this_minute = 0

    def _build_search_queries(
        self,
        company_name: str,
        culture_war_event: str = None,
        include_insider_trading: bool = True
    ) -> List[str]:
        """
        Build search queries for news articles about culture war events and insider trading.

        Args:
            company_name: Full company name
            culture_war_event: Description of the culture war event
            include_insider_trading: Whether to include insider trading queries

        Returns:
            List of search query strings
        """
        queries = [company_name]

        if culture_war_event:
            # Extract key terms from the culture war event
            event_lower = culture_war_event.lower()

            # Add the full company + event query
            queries.append(f"{company_name} {culture_war_event[:50]}")

            # Add specific keyword-based queries
            if 'boycott' in event_lower:
                queries.append(f"{company_name} boycott")
            if 'pride' in event_lower or 'lgbtq' in event_lower or 'trans' in event_lower:
                queries.append(f"{company_name} LGBTQ")
                queries.append(f"{company_name} Pride transgender")
            if 'backlash' in event_lower:
                queries.append(f"{company_name} backlash")
            if 'controversy' in event_lower or 'controversial' in event_lower:
                queries.append(f"{company_name} controversy")
            if 'campaign' in event_lower or 'ad' in event_lower:
                queries.append(f"{company_name} advertisement controversy")
            if 'racist' in event_lower or 'racial' in event_lower or 'race' in event_lower:
                queries.append(f"{company_name} racism")
            if 'political' in event_lower or 'conservative' in event_lower or 'liberal' in event_lower:
                queries.append(f"{company_name} political")
            if 'kaepernick' in event_lower:
                queries.append(f"{company_name} Kaepernick")
            if 'dylan mulvaney' in event_lower:
                queries.append(f"{company_name} Dylan Mulvaney")

        # Add insider trading queries
        if include_insider_trading:
            queries.append(f"{company_name} insider trading")
            queries.append(f"{company_name} executive stock sales")
            queries.append(f"{company_name} SEC filing insider")

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries[:8]  # Limit to 8 queries

    def search_guardian(
        self,
        ticker: str,
        company_name: str,
        culture_war_event: str = None,
        event_date: datetime = None,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 200
    ) -> List[NewsArticle]:
        """
        Search The Guardian API for articles about culture war events.

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
            culture_war_event: Description of the culture war event
            event_date: Date of the culture war event (for reference)
            start_date: Start date for search
            end_date: End date for search
            max_results: Maximum number of results

        Returns:
            List of NewsArticle objects
        """
        articles = []

        if not self.guardian_api_key:
            logger.warning("Guardian API key not provided. Skipping Guardian search.")
            return articles

        # Set date range
        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 12, 31)

        # Build search queries
        search_queries = self._build_search_queries(company_name, culture_war_event)

        logger.info("  Guardian: Searching for %s with %d queries", ticker, len(search_queries))
        logger.info("    Date range: %s to %s", start_date.date(), end_date.date())

        try:
            for query in search_queries:
                page = 1
                total_pages = 1
                query_articles = 0
                max_per_query = max_results // len(search_queries)

                while page <= total_pages and query_articles < max_per_query:
                    url = "https://content.guardianapis.com/search"
                    params = {
                        'q': query,
                        'from-date': start_date.strftime('%Y-%m-%d'),
                        'to-date': end_date.strftime('%Y-%m-%d'),
                        'page': page,
                        'page-size': 50,
                        'show-fields': 'headline,trailText,wordcount,byline,bodyText',
                        'show-tags': 'all',
                        'api-key': self.guardian_api_key
                    }

                    try:
                        response = requests.get(url, params=params, timeout=15)
                        response.raise_for_status()
                        data = response.json()

                        if data['response']['status'] == 'ok':
                            total_pages = min(data['response']['pages'], 3)

                            for item in data['response']['results']:
                                try:
                                    pub_date = datetime.strptime(
                                        item['webPublicationDate'],
                                        '%Y-%m-%dT%H:%M:%SZ'
                                    )

                                    fields = item.get('fields', {})

                                    article = NewsArticle(
                                        ticker=ticker,
                                        company_name=company_name,
                                        source='The Guardian',
                                        title=fields.get('headline', item['webTitle']),
                                        url=item['webUrl'],
                                        published_date=pub_date,
                                        snippet=fields.get('trailText', ''),
                                        culture_war_event=culture_war_event,
                                        event_date=event_date,
                                        search_query=query,
                                        author=fields.get('byline'),
                                        section=item.get('sectionName'),
                                        word_count=fields.get('wordcount'),
                                        body_text=fields.get('bodyText', '')
                                    )
                                    articles.append(article)
                                    query_articles += 1

                                except Exception as e:
                                    logger.debug("Error parsing Guardian article: %s", e)
                                    continue

                        page += 1
                        time.sleep(0.2)

                    except requests.exceptions.RequestException as e:
                        logger.warning("Error fetching Guardian page %d: %s", page, e)
                        time.sleep(2)
                        break

                logger.info("    Query '%.40s...': %d articles", query, query_articles)
                time.sleep(0.3)

        except Exception as e:
            logger.error("Error in Guardian search for %s: %s", ticker, e)

        logger.info("  Guardian total: %d articles", len(articles))
        return articles

    def search_nyt(
        self,
        ticker: str,
        company_name: str,
        culture_war_event: str = None,
        event_date: datetime = None,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 200
    ) -> List[NewsArticle]:
        """
        Search New York Times Article Search API for culture war event articles.

        NYT API rate limit: 500 requests per day, 5 requests per minute

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
            culture_war_event: Description of the culture war event
            event_date: Date of the culture war event (for reference)
            start_date: Start date for search
            end_date: End date for search
            max_results: Maximum number of results

        Returns:
            List of NewsArticle objects
        """
        articles = []

        if not self.nyt_api_key:
            logger.warning("NYT API key not provided. Skipping NYT search.")
            return articles

        # Set date range
        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 12, 31)

        # Build search queries
        search_queries = self._build_search_queries(company_name, culture_war_event)

        logger.info("  NYT: Searching for %s with %d queries", ticker, len(search_queries))
        logger.info("    Date range: %s to %s", start_date.date(), end_date.date())

        try:
            for query in search_queries:
                page = 0
                query_articles = 0
                max_per_query = max_results // len(search_queries)

                while page < 10 and query_articles < max_per_query:
                    self._nyt_rate_limit()

                    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
                    params = {
                        'q': query,
                        'begin_date': start_date.strftime('%Y%m%d'),
                        'end_date': end_date.strftime('%Y%m%d'),
                        'page': page,
                        'api-key': self.nyt_api_key,
                        'sort': 'relevance'
                    }

                    try:
                        response = requests.get(url, params=params, timeout=15)

                        if response.status_code == 429:
                            logger.warning("NYT rate limit hit, waiting 60 seconds...")
                            time.sleep(60)
                            continue

                        response.raise_for_status()
                        data = response.json()

                        if data['status'] == 'OK':
                            docs = data['response']['docs']

                            if not docs:
                                break

                            for doc in docs:
                                try:
                                    pub_date = datetime.strptime(
                                        doc['pub_date'],
                                        '%Y-%m-%dT%H:%M:%S%z'
                                    ).replace(tzinfo=None)

                                    author = None
                                    if doc.get('byline', {}).get('original'):
                                        author = doc['byline']['original']

                                    article = NewsArticle(
                                        ticker=ticker,
                                        company_name=company_name,
                                        source='New York Times',
                                        title=doc.get('headline', {}).get('main', ''),
                                        url=doc.get('web_url', ''),
                                        published_date=pub_date,
                                        snippet=doc.get('snippet', ''),
                                        culture_war_event=culture_war_event,
                                        event_date=event_date,
                                        search_query=query,
                                        author=author,
                                        section=doc.get('section_name'),
                                        word_count=doc.get('word_count'),
                                        body_text=doc.get('lead_paragraph', '')
                                    )
                                    articles.append(article)
                                    query_articles += 1

                                except Exception as e:
                                    logger.debug("Error parsing NYT article: %s", e)
                                    continue

                        page += 1

                    except requests.exceptions.RequestException as e:
                        logger.warning("Error fetching NYT page %d: %s", page, e)
                        time.sleep(5)
                        break

                logger.info("    Query '%.40s...': %d articles", query, query_articles)
                time.sleep(1)

        except Exception as e:
            logger.error("Error in NYT search for %s: %s", ticker, e)

        logger.info("  NYT total: %d articles", len(articles))
        return articles

    def _nyt_rate_limit(self):
        """Enforce NYT API rate limit: 5 requests per minute."""
        now = datetime.now()

        if self.last_nyt_request:
            time_diff = (now - self.last_nyt_request).total_seconds()

            if time_diff < 60:
                if self.nyt_requests_this_minute >= 5:
                    sleep_time = 60 - time_diff + 1
                    logger.info("NYT rate limit: sleeping %.1fs", sleep_time)
                    time.sleep(sleep_time)
                    self.nyt_requests_this_minute = 0
            else:
                self.nyt_requests_this_minute = 0

        self.last_nyt_request = datetime.now()
        self.nyt_requests_this_minute += 1

    def search_reddit(
        self,
        ticker: str,
        company_name: str,
        culture_war_event: str = None,
        event_date: datetime = None,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 200,
        subreddits: List[str] = None
    ) -> List[NewsArticle]:
        """
        Search Reddit for company culture war event mentions.

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
            culture_war_event: Description of the culture war event
            event_date: Date of the culture war event (for reference)
            start_date: Start date for search
            end_date: End date for search
            max_results: Maximum number of results per subreddit
            subreddits: List of subreddits to search

        Returns:
            List of NewsArticle objects
        """
        articles = []

        if not self.reddit:
            logger.warning("Reddit client not initialized. Skipping Reddit search.")
            return articles

        # Set date range
        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 12, 31)

        if subreddits is None:
            subreddits = [
                'news', 'business', 'investing', 'stocks', 'wallstreetbets',
                'finance', 'economy', 'worldnews', 'politics', 'technology',
                'entertainment', 'Conservative', 'progressive', 'capitalism',
                'OutOfTheLoop', 'nottheonion'
            ]

        # Build search queries using the helper method
        search_queries = self._build_search_queries(company_name, culture_war_event)

        logger.info("  Reddit: Searching for %s with %d queries", ticker, len(search_queries))
        logger.info("    Date range: %s to %s", start_date.date(), end_date.date())

        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    for query in search_queries:
                        try:
                            for submission in subreddit.search(
                                query,
                                time_filter='all',
                                limit=max_results // len(search_queries),
                                sort='relevance'
                            ):
                                created = datetime.fromtimestamp(submission.created_utc)

                                if created < start_date or created > end_date:
                                    continue

                                article = NewsArticle(
                                    ticker=ticker,
                                    company_name=company_name,
                                    source=f'Reddit r/{subreddit_name}',
                                    title=submission.title,
                                    url=f"https://reddit.com{submission.permalink}",
                                    published_date=created,
                                    snippet=(
                                        submission.selftext[:500]
                                        if submission.selftext
                                        else None
                                    ),
                                    culture_war_event=culture_war_event,
                                    event_date=event_date,
                                    search_query=query,
                                    author=(
                                        str(submission.author)
                                        if submission.author
                                        else None
                                    ),
                                    body_text=(
                                        submission.selftext
                                        if submission.selftext
                                        else None
                                    )
                                )
                                articles.append(article)

                            time.sleep(0.5)

                        except Exception as e:
                            logger.debug(
                                f"Error searching '{query}' in r/{subreddit_name}: {e}"
                            )
                            continue

                    time.sleep(1)

                except Exception as e:
                    logger.debug("Error accessing r/%s: %s", subreddit_name, e)
                    continue

        except Exception as e:
            logger.error("Error in Reddit search for %s: %s", ticker, e)

        logger.info("  Reddit total: %d posts", len(articles))
        return articles

    def fetch_article_text(self, url: str) -> Optional[str]:
        """
        Fetch full article text from a URL using BeautifulSoup.

        Scrapes the article body from the page HTML. Looks for common
        article body selectors (<article>, .article-body, .story-body,
        .content-body, etc.) and falls back to <p> tags within the page.

        Args:
            url: The article URL to fetch.

        Returns:
            Cleaned article body text, or None on failure.
        """
        try:
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script, style, nav, header, footer elements
            for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()

            # Try common article body selectors in order of specificity
            body_element = None
            selectors = [
                'article [class*="article-body"]',
                'article [class*="story-body"]',
                'article [class*="content-body"]',
                '[class*="article-body"]',
                '[class*="story-body"]',
                '[class*="content-body"]',
                '[class*="article__body"]',
                '[class*="story__body"]',
                '[class*="post-content"]',
                '[class*="entry-content"]',
                '[itemprop="articleBody"]',
                'article',
            ]

            for selector in selectors:
                body_element = soup.select_one(selector)
                if body_element:
                    break

            if body_element:
                # Extract text from paragraphs within the body element
                paragraphs = body_element.find_all('p')
                if paragraphs:
                    text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs)
                else:
                    text = body_element.get_text(separator='\n', strip=True)
            else:
                # Fallback: collect all <p> tags from the page
                paragraphs = soup.find_all('p')
                if not paragraphs:
                    return None
                text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs)

            # Clean whitespace: collapse runs of whitespace, strip lines
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

            # Only return if we got meaningful content (more than a sentence)
            if len(text) < 50:
                return None

            return text

        except requests.exceptions.RequestException as e:
            logger.debug("Failed to fetch article text from %s: %s", url, e)
            return None
        except Exception as e:
            logger.debug("Error parsing article text from %s: %s", url, e)
            return None
        finally:
            # Rate limit: 1 request per second
            time.sleep(1)

    def aggregate_culture_war_news(
        self,
        culture_war_df: pd.DataFrame,
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        max_results_per_source: int = 200,
        sources: List[str] = None,
        checkpoint_file: str = 'news_checkpoint.csv'
    ) -> pd.DataFrame:
        """
        Aggregate news from all sources for culture war events.

        Searches for news about each company's culture war event across
        the full date range. Also searches for insider trading news
        related to these companies.

        Args:
            culture_war_df: DataFrame with columns:
                - 'Company': Company name
                - 'Ticker': Stock ticker symbol
                - 'Culture War Event': Description of the event
                - 'Event Date': Date of the event
            start_date: Start date for search (default: '2000-01-01')
            end_date: End date for search (default: '2025-12-31')
            max_results_per_source: Max results per source per event
            sources: List of sources to use ['guardian', 'nyt', 'reddit']
            checkpoint_file: File to save progress

        Returns:
            DataFrame with all aggregated news articles
        """
        if sources is None:
            sources = ['guardian', 'nyt', 'reddit']

        all_articles = []

        checkpoint_path = Path(checkpoint_file)
        processed_events = set()

        if checkpoint_path.exists():
            try:
                checkpoint_df = pd.read_csv(checkpoint_path)
                checkpoint_df['published_date'] = pd.to_datetime(
                    checkpoint_df['published_date']
                )
                all_articles.extend(checkpoint_df.to_dict('records'))

                # Track processed events by ticker + event_date
                for _, row in checkpoint_df.iterrows():
                    ticker = row.get('ticker', '')
                    event_date = row.get('event_date', '')
                    if ticker and event_date:
                        processed_events.add(f"{ticker}_{event_date}")

                logger.info("Loaded checkpoint with %d articles", len(checkpoint_df))
                logger.info("Processed events: %d", len(processed_events))
            except Exception as e:
                logger.warning("Error loading checkpoint: %s", e)

        total_events = len(culture_war_df)

        for idx, row in culture_war_df.iterrows():
            # Extract event details
            company_name = row.get('Company', '')
            ticker = row.get('Ticker', '')
            culture_war_event = row.get('Culture War Event', '')
            event_date_raw = row.get('Event Date', None)

            # Skip if no ticker
            if not ticker or pd.isna(ticker) or ticker in ['Private', 'N/A']:
                logger.info("Skipping %s - no valid ticker", company_name)
                continue

            # Parse event date
            event_date = None
            if event_date_raw and not pd.isna(event_date_raw):
                try:
                    event_date = pd.to_datetime(event_date_raw)
                except Exception:
                    logger.warning("Could not parse event date: %s", event_date_raw)

            # Check if already processed
            event_key = f"{ticker}_{event_date}"
            if event_key in processed_events:
                logger.info("Skipping %s - already processed", ticker)
                continue

            logger.info("=" * 60)
            logger.info("[%d/%d] %s (%s)", idx + 1, total_events, company_name, ticker)
            logger.info("Event: %.80s...", culture_war_event)
            if event_date:
                logger.info("Event Date: %s", event_date.date())
            logger.info("=" * 60)

            event_articles = []

            # Parse date range
            search_start = datetime.strptime(start_date, '%Y-%m-%d')
            search_end = datetime.strptime(end_date, '%Y-%m-%d')

            # Search Guardian
            if 'guardian' in sources:
                logger.info("Searching The Guardian...")
                articles = self.search_guardian(
                    ticker=ticker,
                    company_name=company_name,
                    culture_war_event=culture_war_event,
                    event_date=event_date,
                    start_date=search_start,
                    end_date=search_end,
                    max_results=max_results_per_source
                )
                event_articles.extend(articles)

            # Search NYT
            if 'nyt' in sources:
                logger.info("Searching New York Times...")
                articles = self.search_nyt(
                    ticker=ticker,
                    company_name=company_name,
                    culture_war_event=culture_war_event,
                    event_date=event_date,
                    start_date=search_start,
                    end_date=search_end,
                    max_results=max_results_per_source
                )
                event_articles.extend(articles)

            # Search Reddit
            if 'reddit' in sources:
                logger.info("Searching Reddit...")
                articles = self.search_reddit(
                    ticker=ticker,
                    company_name=company_name,
                    culture_war_event=culture_war_event,
                    event_date=event_date,
                    start_date=search_start,
                    end_date=search_end,
                    max_results=max_results_per_source
                )
                event_articles.extend(articles)

            # Save progress
            if event_articles:
                all_articles.extend([vars(a) for a in event_articles])
                checkpoint_df = pd.DataFrame(all_articles)
                checkpoint_df.to_csv(checkpoint_path, index=False)
                logger.info("Checkpoint saved: %d total articles", len(all_articles))

            processed_events.add(event_key)
            time.sleep(2)

        if len(all_articles) > 0:
            df = pd.DataFrame(all_articles)

            original_len = len(df)
            df = df.drop_duplicates(subset=['url'], keep='first')
            logger.info("Removed %d duplicate articles", original_len - len(df))

            df['published_date'] = pd.to_datetime(df['published_date'])
            df = df.sort_values('published_date', ascending=False)
        else:
            df = pd.DataFrame()

        return df

    def save_news(self, news_df: pd.DataFrame, output_path: str):
        """Save news articles to CSV with summary statistics."""
        if len(news_df) > 0:
            news_df.to_csv(output_path, index=False)
            logger.info("=" * 60)
            logger.info("SAVED: %d articles to %s", len(news_df), output_path)
            logger.info("=" * 60)

            logger.info("=== SUMMARY STATISTICS ===")
            logger.info("Total articles: %d", len(news_df))
            logger.info(
                "Date range: %s to %s",
                news_df['published_date'].min().date(),
                news_df['published_date'].max().date()
            )
            logger.info("Unique companies: %d", news_df['ticker'].nunique())

            logger.info("--- Articles by Source ---")
            for source, count in news_df['source'].value_counts().items():
                logger.info("  %s: %d", source, count)

            logger.info("--- Top 10 Companies by Article Count ---")
            for ticker, count in news_df['ticker'].value_counts().head(10).items():
                logger.info("  %s: %d", ticker, count)

            logger.info("--- Articles by Year ---")
            yearly = news_df['published_date'].dt.year.value_counts().sort_index()
            for year, count in yearly.items():
                logger.info("  %s: %d", year, count)
        else:
            logger.warning("No articles to save")



def load_news_data(
    cache_file='./news_data/culture_war_news_2000_2025_final.csv',
    refresh=False,
    sources=None
):
    """
    Load news articles data from Guardian, NYT, and Reddit.

    Parameters:
    -----------
    cache_file : str
        Path to cached news CSV file
    refresh : bool
        If True, re-download data even if cache exists
    sources : list
        List of news sources to include ['guardian', 'nyt', 'reddit']

    Returns:
    --------
    pd.DataFrame : News articles
    """
    if sources is None:
        sources = ['guardian', 'nyt', 'reddit']

    if os.path.exists(cache_file) and not refresh:
        logger.info("Loading cached news data from %s", cache_file)
        news_df = pd.read_csv(cache_file)
        news_df['published_date'] = pd.to_datetime(news_df['published_date'])

        if sources:
            source_patterns = {
                'guardian': 'The Guardian',
                'nyt': 'New York Times',
                'reddit': 'Reddit r/'
            }
            keep_sources = []
            for src in sources:
                if src in source_patterns:
                    keep_sources.append(source_patterns[src])

            if keep_sources:
                news_df = news_df[
                    news_df['source'].str.contains(
                        '|'.join(keep_sources), case=False, na=False
                    )
                ]

        return news_df
    else:
        logger.warning("News data not found at %s", cache_file)
        logger.warning("Run scrape_culture_war_news() to generate this data")
        return None


def scrape_culture_war_news(
    culture_war_csv: str = 'Culture_War_Companies_160_fullmeta.csv',
    output_file: str = './news_data/culture_war_news.csv',
    start_date: str = '2000-01-01',
    end_date: str = '2025-12-31',
    max_results_per_source: int = 200,
    sources: List[str] = None
) -> pd.DataFrame:
    """
    Scrape news about culture war companies and their events.

    This function searches for news articles about each company's culture war
    event across the full date range. It also searches for insider trading
    news related to these companies.

    Parameters:
    -----------
    culture_war_csv : str
        Path to the culture war companies CSV file
    output_file : str
        Path to save the output CSV file
    start_date : str
        Start date for search (default: '2000-01-01')
    end_date : str
        End date for search (default: '2025-12-31')
    max_results_per_source : int
        Maximum articles per source per event (default: 200)
    sources : list
        List of sources to use ['guardian', 'nyt', 'reddit']

    Returns:
    --------
    pd.DataFrame : DataFrame with all scraped news articles

    Note:
    -----
    Requires API keys to be set in environment variables or .env file:
    - GUARDIAN_API_KEY: For The Guardian API
    - NYT_API_KEY: For New York Times API
    - REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT: For Reddit API
    """
    if sources is None:
        sources = ['guardian', 'nyt', 'reddit']

    # Load culture war data
    logger.info("Loading culture war companies data...")
    culture_war_df = import_culture_war_data(culture_war_csv)
    logger.info("Loaded %d culture war events", len(culture_war_df))

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize news aggregator
    logger.info("Initializing news aggregator...")
    aggregator = CompanyNewsAggregator(
        guardian_api_key=os.getenv('GUARDIAN_API_KEY'),
        nyt_api_key=os.getenv('NYT_API_KEY'),
        reddit_client_id=os.getenv('REDDIT_CLIENT_ID'),
        reddit_client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        reddit_user_agent=os.getenv('REDDIT_USER_AGENT', 'CultureWarResearch/1.0')
    )

    # Check which APIs are available
    available_sources = []
    if aggregator.guardian_api_key:
        available_sources.append('guardian')
        logger.info("  Guardian API: Available")
    else:
        logger.info("  Guardian API: Not configured (set GUARDIAN_API_KEY)")

    if aggregator.nyt_api_key:
        available_sources.append('nyt')
        logger.info("  NYT API: Available")
    else:
        logger.info("  NYT API: Not configured (set NYT_API_KEY)")

    if aggregator.reddit:
        available_sources.append('reddit')
        logger.info("  Reddit API: Available")
    else:
        logger.info("  Reddit API: Not configured (set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)")

    # Filter sources to only available ones
    sources = [s for s in sources if s in available_sources]

    if not sources:
        logger.error("No API keys configured. Please set at least one of:")
        logger.error("  - GUARDIAN_API_KEY")
        logger.error("  - NYT_API_KEY")
        logger.error("  - REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET")
        return pd.DataFrame()

    logger.info("Using sources: %s", sources)
    logger.info("Date range: %s to %s", start_date, end_date)
    logger.info("Max results per source: %d", max_results_per_source)

    # Run the aggregation
    logger.info("=" * 60)
    logger.info("Starting news scraping...")
    logger.info("=" * 60)

    news_df = aggregator.aggregate_culture_war_news(
        culture_war_df=culture_war_df,
        start_date=start_date,
        end_date=end_date,
        max_results_per_source=max_results_per_source,
        sources=sources,
        checkpoint_file=output_file.replace('.csv', '_checkpoint.csv')
    )

    # Save final results
    if len(news_df) > 0:
        aggregator.save_news(news_df, output_file)
    else:
        logger.warning("No articles found. Check API keys and try again.")

    return news_df


def enrich_news_with_text(
    news_csv: str = './news_data/culture_war_news.csv',
    output_csv: str = './news_data/culture_war_news_fulltext.csv',
    checkpoint_interval: int = 100,
    max_per_ticker: int = 50,
) -> pd.DataFrame:
    """
    Enrich existing news CSV with full article body text.

    Fetches article text from URLs for articles that don't have body_text.
    Saves checkpoint every checkpoint_interval articles.
    Limits to max_per_ticker articles per company (most relevant first).

    Args:
        news_csv: Path to the input news CSV file.
        output_csv: Path to save the enriched output CSV file.
        checkpoint_interval: Save checkpoint every N articles fetched.
        max_per_ticker: Maximum articles to enrich per ticker symbol.

    Returns:
        DataFrame with body_text column populated where possible.
    """
    logger.info("Loading news data from %s", news_csv)
    df = pd.read_csv(news_csv)
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    logger.info("Loaded %d articles", len(df))

    # Add body_text column if it doesn't exist
    if 'body_text' not in df.columns:
        df['body_text'] = None

    # If output file exists, load it to resume from checkpoint
    output_path = Path(output_csv)
    if output_path.exists():
        logger.info("Loading existing output for resume: %s", output_csv)
        existing = pd.read_csv(output_csv)
        existing['published_date'] = pd.to_datetime(
            existing['published_date'], errors='coerce'
        )
        # Merge body_text from existing output into current df
        if 'body_text' in existing.columns:
            existing_text = existing.set_index('url')['body_text'].dropna().to_dict()
            for url, text in existing_text.items():
                mask = df['url'] == url
                df.loc[mask, 'body_text'] = text
            logger.info(
                "Resumed with %d articles already having body_text",
                df['body_text'].notna().sum()
            )

    # Identify articles that need body_text fetching
    needs_text = df[
        df['body_text'].isna() | (df['body_text'] == '')
    ].copy()
    logger.info("Articles needing body_text: %d", len(needs_text))

    # Limit per ticker: prioritize by source reliability
    # Guardian > NYT > Reddit (Guardian API already gives body, so those
    # should already be populated; focus on NYT and Reddit URLs)
    source_priority = {'The Guardian': 0, 'New York Times': 1}
    needs_text['_source_priority'] = needs_text['source'].map(
        lambda s: source_priority.get(s, 2)
    )
    needs_text = needs_text.sort_values('_source_priority')

    # Limit per ticker
    selected = needs_text.groupby('ticker').head(max_per_ticker)
    logger.info(
        "Selected %d articles to fetch (%d tickers, max %d each)",
        len(selected), selected['ticker'].nunique(), max_per_ticker
    )

    # Initialize aggregator for fetch_article_text method
    aggregator = CompanyNewsAggregator()

    fetched_count = 0
    success_count = 0

    for idx, row in selected.iterrows():
        url = row.get('url', '')
        if not url or pd.isna(url):
            continue

        # Skip Reddit self-posts (no article to scrape)
        if 'reddit.com' in str(url):
            continue

        logger.debug("Fetching [%d/%d]: %s", fetched_count + 1, len(selected), url)
        text = aggregator.fetch_article_text(url)

        if text:
            df.at[idx, 'body_text'] = text
            success_count += 1

        fetched_count += 1

        # Checkpoint
        if fetched_count % checkpoint_interval == 0:
            df.to_csv(output_csv, index=False)
            logger.info(
                "Checkpoint: %d/%d fetched, %d successful, saved to %s",
                fetched_count, len(selected), success_count, output_csv
            )

    # Final save
    df.to_csv(output_csv, index=False)

    total_with_text = df['body_text'].notna().sum()
    total_empty = df['body_text'].isna().sum()
    logger.info("=" * 60)
    logger.info("Enrichment complete.")
    logger.info("  Total articles: %d", len(df))
    logger.info("  With body_text: %d (%.1f%%)", total_with_text, 100 * total_with_text / len(df))
    logger.info("  Without body_text: %d", total_empty)
    logger.info("  Newly fetched: %d attempted, %d successful", fetched_count, success_count)
    logger.info("  Saved to: %s", output_csv)
    logger.info("=" * 60)

    return df
