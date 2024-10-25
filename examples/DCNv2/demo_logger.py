import logging

def get_logger(log_level: str):
    dcnv2_logger = logging.getLogger("DCNv2")
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    dcnv2_logger.addHandler(stream_handler)
    dcnv2_logger.setLevel(log_level)
    return dcnv2_logger


logger = get_logger("Debug")