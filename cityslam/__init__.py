import os
import logging

# Fix default file creation permissions
os.umask(0o002)

formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("cityslam")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
