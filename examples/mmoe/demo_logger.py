import logging

def get_logger(log_level: str):
    mmoe_logger = logging.getLogger("mmoe")
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    mmoe_logger.addHandler(stream_handler)
    mmoe_logger.setLevel(log_level)
    return mmoe_logger


logger = get_logger("DEBUG")