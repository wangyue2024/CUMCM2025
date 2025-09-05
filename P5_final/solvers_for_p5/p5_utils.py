# solvers_for_p5/p5_utils.py

import logging
import os
from .p5_config import P5Config

def setup_logger():
    """配置日志记录器，同时输出到控制台和文件。"""
    os.makedirs(P5Config.RESULTS_DIR, exist_ok=True)
    
    logger = logging.getLogger("Problem5Solver")
    logger.setLevel(logging.INFO)
    
    # 防止重复添加handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器
    fh = logging.FileHandler(P5Config.LOG_FILE, mode='a')
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger