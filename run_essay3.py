"""Run Essay 3 pipeline: Insider Trading analysis."""
import sys
import os
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from model.essay3 import run_essay3

print("=" * 60)
print("  Essay 3: Insider Trading Analysis")
print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

result = run_essay3()

print(f"\nEssay 3 completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
