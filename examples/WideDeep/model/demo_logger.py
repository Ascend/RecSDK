import logging

def get_logger(log_level: str):
    wide_deep_logger = logging.getLogger("wide_deep")
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    wide_deep_logger.addHandler(stream_handler)
    wide_deep_logger.setLevel(log_level)
    return wide_deep_logger


logger = get_logger("Debug")