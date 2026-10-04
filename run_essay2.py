"""Run Essay 2 pipeline: NLP analysis, political alignment, and DiD."""
import sys
import os
import logging
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from model.datastore import DataStore
from model.essay1 import estimate_vix_regimes, sentiment_by_regime
from model.essay2 import run_nlp_analysis, compute_political_alignment, save_nlp_results, save_alignment_results
from model.essay2_did import run_did, save_did_results

print("=" * 60)
print("  Essay 2: Event Study + NLP + DiD")
print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

store = DataStore()

# Step 1: NLP Analysis (FinBERT scoring of news + filings)
print("\nStep 1: Running NLP analysis...")
try:
    nlp_result = run_nlp_analysis(store, download_filings=False)
    if nlp_result is not None:
        print(f"  Scored {nlp_result.n_articles_scored} articles")
        save_nlp_results(store, nlp_result)
        print("  NLP results saved.")
    else:
        print("  NLP analysis returned None (may lack news data)")
except Exception as e:
    print(f"  NLP analysis error: {e}")
    nlp_result = None

# Step 2: Political alignment
print("\nStep 2: Computing political alignment...")
try:
    alignment_result = compute_political_alignment(store)
    if alignment_result is not None:
        print(f"  Aligned {len(alignment_result.company_scores)} companies")
        save_alignment_results(store, alignment_result)
        print("  Alignment results saved.")
    else:
        print("  Alignment returned None")
except Exception as e:
    print(f"  Alignment error: {e}")

# Step 3: DiD
print("\nStep 3: Running DiD analysis...")
try:
    regime_result = estimate_vix_regimes(store, n_regimes=3)
    sentiment_analysis = sentiment_by_regime(store, regime_result=regime_result)
    did_result = run_did(store, regime_result=regime_result, sentiment_analysis=sentiment_analysis)
    if did_result is not None:
        print(f"  DiD specs: {len(did_result.specifications)}")
        save_did_results(store, did_result)
        print("  DiD results saved.")
    else:
        print("  DiD returned None")
except Exception as e:
    print(f"  DiD error: {e}")

store.close()
print(f"\nEssay 2 completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
