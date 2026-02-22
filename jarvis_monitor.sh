#!/bin/bash
# Jarvis News Scraping Monitor
# Email: aroseboro@roseboroholdings.com
# Sends notifications via Jarvis API

JARVIS_API="http://207.254.38.26:8081"
CHECKPOINT_FILE="/Users/administrator/signalsandsystems/news_data/culture_war_news_checkpoint.csv"
CHECK_INTERVAL=600  # Check every 10 minutes
EMAIL="aroseboro@roseboroholdings.com"

send_jarvis_notification() {
    local message="$1"
    # Use Jarvis chat to send email and update dashboard
    curl -s --max-time 120 -X POST "$JARVIS_API/chat" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$message\"}" \
        > /dev/null 2>&1
}

is_running() {
    pgrep -f "scrape_culture_war_news" > /dev/null 2>&1
    return $?
}

get_article_count() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        wc -l < "$CHECKPOINT_FILE" | tr -d ' '
    else
        echo "0"
    fi
}

echo "================================================"
echo "Jarvis News Scraping Monitor"
echo "================================================"
echo "Email: $EMAIL"
echo "Check interval: $((CHECK_INTERVAL/60)) minutes"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "================================================"
echo ""

updates_sent=0

while true; do
    current_count=$(get_article_count)
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    if is_running; then
        echo "[$timestamp] RUNNING | Articles: $current_count"

        # Send progress update every 30 minutes (3 checks)
        updates_sent=$((updates_sent + 1))
        if [ $((updates_sent % 3)) -eq 0 ]; then
            echo "[$timestamp] Sending progress update to $EMAIL..."
            send_jarvis_notification "Send an email to $EMAIL with subject: News Scraping Progress - $current_count articles. Body: Automated update from your news scraping monitor. Progress: $current_count articles collected from The Guardian and Reddit. Status: Still running. Next update in 30 minutes."
            echo "[$timestamp] Update sent."
        fi
    else
        echo "[$timestamp] COMPLETED | Total: $current_count articles"
        echo ""
        echo "Sending completion notification..."

        send_jarvis_notification "URGENT: Send an email to $EMAIL with subject: NEWS SCRAPING COMPLETE - $current_count articles collected! Body: Great news! Your culture war news scraping job has finished successfully. Total articles: $current_count. Data file: $CHECKPOINT_FILE. Sources: The Guardian API and Reddit API. You can now proceed with data analysis for your dissertation research. This is fantastic progress for Article 1!"

        echo ""
        echo "================================================"
        echo "        SCRAPING COMPLETE!"
        echo "================================================"
        echo "Total articles: $current_count"
        echo "File: $CHECKPOINT_FILE"
        echo "Notification sent to: $EMAIL"
        echo "================================================"

        exit 0
    fi

    sleep $CHECK_INTERVAL
done
