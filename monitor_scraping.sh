#!/bin/bash
# News Scraping Monitor Script for Jarvis
# Checks progress and outputs status for notifications

CHECKPOINT_FILE="/Users/administrator/signalsandsystems/news_data/culture_war_news_checkpoint.csv"
LOG_FILE="/private/tmp/claude/-Users-administrator/tasks/b94b44c.output"
TOTAL_COMPANIES=160

# Check if scraping process is still running
is_running() {
    pgrep -f "scrape_culture_war_news" > /dev/null 2>&1
    return $?
}

# Get current article count
get_article_count() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        wc -l < "$CHECKPOINT_FILE" | tr -d ' '
    else
        echo "0"
    fi
}

# Get last processed company from log
get_last_company() {
    if [ -f "$LOG_FILE" ]; then
        grep -E "^\[.*\] .* \(" "$LOG_FILE" | tail -1 | sed 's/.*\[\([0-9]*\)\/160\].*/\1/'
    else
        echo "0"
    fi
}

# Get latest checkpoint info
get_latest_checkpoint() {
    if [ -f "$LOG_FILE" ]; then
        grep "Checkpoint saved" "$LOG_FILE" | tail -1
    else
        echo "No checkpoint data"
    fi
}

# Main status check
ARTICLES=$(get_article_count)
RUNNING=$(is_running && echo "true" || echo "false")

# Get company progress from checkpoint file modification
if [ -f "$CHECKPOINT_FILE" ]; then
    LAST_MODIFIED=$(stat -f "%Sm" "$CHECKPOINT_FILE")
else
    LAST_MODIFIED="N/A"
fi

# Output JSON-formatted status for Jarvis
echo "{"
echo "  \"status\": \"$([ "$RUNNING" = "true" ] && echo "running" || echo "completed")\","
echo "  \"articles_collected\": $ARTICLES,"
echo "  \"checkpoint_file\": \"$CHECKPOINT_FILE\","
echo "  \"last_updated\": \"$LAST_MODIFIED\","
echo "  \"process_running\": $RUNNING"
echo "}"

# Human-readable summary
echo ""
echo "=== Scraping Status ==="
if [ "$RUNNING" = "true" ]; then
    echo "Status: RUNNING"
else
    echo "Status: COMPLETED"
fi
echo "Articles: $ARTICLES"
echo "Last Updated: $LAST_MODIFIED"

# Exit with appropriate code
if [ "$RUNNING" = "true" ]; then
    exit 0  # Still running
else
    exit 100  # Completed (custom exit code for Jarvis)
fi
