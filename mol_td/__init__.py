from .utils import *
from .model import *

models = {'SimpleVAE': SimpleTDVAE,
          'SimpleTDVAE': SimpleTDVAE}

'''
check_types depreciated warning from tfd described here https://github.com/tensorflow/probability/issues/1523
suppressed with this code
'''
import logging
logger = logging.getLogger("root")
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()
logger.addFilter(CheckTypesFilter())