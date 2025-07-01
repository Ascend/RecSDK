"""define logging configure"""
from npu_bridge.npu_init import *
import logging
from datetime import datetime, timedelta, timezone
import platform

__all__ = ["Log"]
class Log(object):
    def __init__(self, ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                                      datefmt="%m/%d/%Y %H:%M:%S %p")
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

