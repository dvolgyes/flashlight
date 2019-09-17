#!/usr/bin/env python3

# Flashlight.auto

from loguru import logger
import yaml
import dpath
from ..env import detect_libs, free_space
from flashlight.env import cfg_reader
from flashlight.util import resolve_environment
from flashlight.util import ensure_reproducible_results
from flashlight.engine import Engine
from pathlib import Path
import os
import pyfiglet
from textwrap import indent
import termtables as tt
import sys
from dotenv import load_dotenv
load_dotenv()


def generic_report():
    results = []
    results.append(('Working directory', os.getcwd()))
    results.append(('free space', f'{free_space():.02f}  GB'))
    results.append(('Python', sys.version[0:5]))

    CONDA = os.environ.get('CONDA_DEFAULT_ENV', 'Not in use')
    results.append(('Conda environment', CONDA))

    s = tt.to_string(results)
    logger.info('\n'+indent(f'Base system information:\n{s}', ' '*4))

    detected_libs = []
    for f in ['requirements-conda.txt', 'requirements.txt']:
        reqs = Path(f)
        if reqs.exists():
            with open(reqs, 'rt') as f:
                for line in f:
                    lib = line.strip().split()
                    if len(lib) > 0:
                        lib = lib[0]
                        if lib in sys.modules:
                            d = detect_libs(lib)
                            if d['enabled']:
                                detected_libs.append((d['name'], d['version']))
    if len(detected_libs):
        s = tt.to_string(detected_libs, header=['package name', 'version'])
        logger.info(
            '\n'+indent(f'Imported non-standard libraries:\n{s}', ' '*4))


def auto_init(section=None):
    global engine
    logger.success(
        '\n'+indent(pyfiglet.figlet_format('Flashlight', 'cybermedium'), '    '))
    # ~ generic_report()
    cfg = auto_cfg()
    logger.debug('Configuration:\n'+cfg.yaml)
    if section is None:
        engine = Engine(cfg)
    else:
        engine = Engine(cfg[section])
    return engine
    # ~ results = []
    # ~ names = [name.split('.')[0] for name in sys.modules.keys()]
    # ~ for lib in ['numpy', 'torch', 'scipy', 'torchvision']:
    # ~ d = detect_libs(lib)
    # ~ if d['enabled']:
    # ~ results.append( (d['name'],d['version']))
    # ~ if len(results):
    # ~ s = indent(tt.to_string(results),'    ')
    # ~ logger.info(f'Imported libraries:\n{s}')


def auto_cfg():
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        filenames = ('config.yaml', 'local_config.yaml')
    cfg = cfg_reader(filenames)
    cfg = resolve_environment(cfg)
    cfg = ensure_reproducible_results(cfg)
    return cfg
