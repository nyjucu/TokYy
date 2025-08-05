from pathlib import Path

BASE_DIR = Path( __file__ ).resolve().parent

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

RESULTS_DIR = BASE_DIR / "results"

LOSSES_DIR = RESULTS_DIR / "losses"
METRICS_DIR = RESULTS_DIR / "metrics"
LEARNING_RATES_DIR = RESULTS_DIR / "learning_rates"
PREDICTS_DIR = RESULTS_DIR / "predicts"
OTHERS_DIR = RESULTS_DIR / "others"

CHECKPOINTS_DIR.mkdir( parents = True, exist_ok = True)
RESULTS_DIR.mkdir( parents = True, exist_ok = True)
METRICS_DIR.mkdir( parents = True, exist_ok = True)
LOSSES_DIR.mkdir( parents = True, exist_ok = True)
LEARNING_RATES_DIR.mkdir( parents = True, exist_ok = True)
PREDICTS_DIR.mkdir( parents = True, exist_ok = True)
OTHERS_DIR.mkdir( parents = True, exist_ok = True)
