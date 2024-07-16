import os, sys

os.chdir("..")
sys.path.insert(0, os.getenv("ROOT_CONF_DIR"))  # type: ignore
sys.path.insert(0, os.path.abspath('.'))

project = 'Deep Radar'

from conf import *
