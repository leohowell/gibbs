# -*- coding: utf-8 -*-

import logging


logger = logging.getLogger('gibbs')
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
    datefmt='%y%m%d %H:%M:%S',
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
