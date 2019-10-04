#!/usr/bin/env python3
import sys
from loguru import logger
from pathlib import Path


def log(cfg, logdir):
    logger.remove()
    for _, log in cfg.loggers.items():
        if log.target == 'display':
            logger.add(sys.stderr, level=log.level)
        elif log.target == 'file':
            fn = Path(logdir) / log.filename
            logger.add(fn, level=log.level)
    for logname, log in cfg.loggers.items():
        if log.target == 'display':
            logger.debug(f'Logger "{logname}" is added (displayed on stderr).')
        else:
            fn = Path(logdir) / log.filename
            logger.debug(f'Logger "{logname}" is added (saved to: {str(fn)}).')
    return logger
