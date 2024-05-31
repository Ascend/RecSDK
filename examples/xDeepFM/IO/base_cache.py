"""define abstract base class"""
from npu_bridge.npu_init import *
import abc

__all__ = ["BaseCache"]


class BaseCache(object):
    """abstract base class"""

    @abc.abstractmethod
    def write_tfrecord(self, infile, outfile, hparams):
        """Subclass must implement this."""
        pass

