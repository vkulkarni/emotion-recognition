import logging
import os
import configparser

logger = logging.getLogger(__name__)

"""
Read Config file
"""
config = configparser.ConfigParser()
config.read('config.ini')

"""
Set log level
"""
log_level = config.getint('logger', 'log_level')
logging.basicConfig(level=log_level)

"""
set BASE DIR
"""
BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
