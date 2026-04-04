"""Backward-compatible entry point. Import from model package."""
from model import *  # noqa: F401,F403
from model import __all__, __version__, __author__  # noqa: F401


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import logging
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print("Signals & Systems — Model Pipeline")
    print(f"Version {__version__}")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    output = run_and_save(save=True)

    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
