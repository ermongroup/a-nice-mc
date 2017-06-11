import logging


def get_logger(module_name, level=logging.INFO):
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger