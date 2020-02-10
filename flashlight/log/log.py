#!/usr/bin/env python3
import sys
from loguru import logger
from pathlib import Path
import dpath
import dpath.util

def log(cfg, logdir):
    logger.remove()
    for _, log in cfg.loggers.items():
        format = log.get('format', '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')
        if log.target == 'display':
            logger.add(sys.stderr, level=log.level, format=format)
        elif log.target == 'file':
            fn = Path(logdir) / log.filename
            logger.add(fn, level=log.level, format=format)
    for logname, log in cfg.loggers.items():
        if log.target == 'display':
            clog(cfg, 'loggers.internals', 'TRACE', f'Logger "{logname}" is added with loglevel "{log.level}" (displayed on stderr).')
        else:
            fn = Path(logdir) / log.filename
            clog(cfg, 'loggers.internals', 'TRACE', f'Logger "{logname}" is added with loglevel "{log.level}" (saved to: {str(fn)}).')
    return logger


def clog(cfg, key, default, message):
    try:
        log_level = dpath.util.get(cfg.log_levels, key, separator='.')
    except KeyError:
        log_level = default
    logger.log(log_level, message)
