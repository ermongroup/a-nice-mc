import logging
import sys
import os
import numpy as np
import pandas as pd
import errno


def create_logger(module_name, level=logging.INFO):
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter('[{}] [%(levelname)s] %(message)s'.format(module_name))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_ess(ess, path):
    df = pd.DataFrame(np.reshape(ess, [1, -1]))
    df.to_csv('{}/ess.csv'.format(path), mode='a', header=False)


def ensure_directory(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
