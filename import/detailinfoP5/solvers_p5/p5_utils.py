# solvers_p5/p5_utils.py
import logging
import os
from .p5_config import P5Config

def setup_logger():
    # ... (代码与之前相同) ...
    os.makedirs(P5Config.RESULTS_DIR, exist_ok=True)
    logger = logging.getLogger("Problem5Solver")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(P5Config.LOG_FILE, mode='a')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger