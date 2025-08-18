from pathlib import Path

BASE_DIR = Path( __file__ ).resolve().parent

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

RESULTS_DIR = BASE_DIR / "results"

LOSSES_DIR = RESULTS_DIR / "losses"
_LOSSES_DIR = RESULTS_DIR / "_losses"
_LOSSES_TEST_DIR = _LOSSES_DIR / "test"
_LOSSES_VAL_DIR = _LOSSES_DIR / "val"
_LOSSES_TRAIN_DIR = _LOSSES_DIR / "train"
METRICS_DIR = RESULTS_DIR / "metrics"
LEARNING_RATES_DIR = RESULTS_DIR / "learning_rates"
PREDICTS_DIR = RESULTS_DIR / "predicts"
OTHERS_DIR = RESULTS_DIR / "others"
GRAD_DIR = RESULTS_DIR / "grad"

CHECKPOINTS_DIR.mkdir( parents = True, exist_ok = True)
RESULTS_DIR.mkdir( parents = True, exist_ok = True)
METRICS_DIR.mkdir( parents = True, exist_ok = True)
LOSSES_DIR.mkdir( parents = True, exist_ok = True)
_LOSSES_DIR.mkdir( parents = True, exist_ok = True)
_LOSSES_TEST_DIR.mkdir( parents = True, exist_ok = True)
_LOSSES_VAL_DIR.mkdir( parents = True, exist_ok = True)
_LOSSES_TRAIN_DIR.mkdir( parents = True, exist_ok = True)
LEARNING_RATES_DIR.mkdir( parents = True, exist_ok = True)
PREDICTS_DIR.mkdir( parents = True, exist_ok = True)
OTHERS_DIR.mkdir( parents = True, exist_ok = True)
GRAD_DIR.mkdir( parents = True, exist_ok = True)
