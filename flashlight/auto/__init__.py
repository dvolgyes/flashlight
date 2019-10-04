#!/usr/bin/env python3

# Flashlight.auto

from loguru import logger
import flashlight
from flashlight.env import detect_libs, free_space, cfg_reader
from flashlight.models import get_model
from flashlight.util import resolve_environment, ensure_reproducible_results
from flashlight.engine import Engine
from flashlight.optim import get_optimizer, get_scheduler
from flashlight.data import get_dataloaders
from flashlight.loss import LossEvaluator
from flashlight.log import log
from pathlib import Path
import os
import pyfiglet
from textwrap import indent
import termtables as tt
import sys
from box import SBox
from datetime import datetime
from tensorboardX import SummaryWriter


def generic_report():
    results = []
    results.append(('Working directory', os.getcwd()))
    results.append(('free space', f'{free_space():.02f}  GB'))
    results.append(('Python', sys.version[0:5]))

    CONDA = os.environ.get('CONDA_DEFAULT_ENV', 'Not in use')
    results.append(('Conda environment', CONDA))

    s = tt.to_string(results)
    logger.info('\n' + indent(f'Base system information:\n{s}', '     '))

    detected_libs = []
    for lib in {lib.split('.')[0] for lib in sys.modules} - {'sys'} - {'flashlight'}:
        if len(lib) > 0:
            try:
                d = detect_libs(lib)
                if d['enabled']:
                    detected_libs.append((d['name'], d['version']))
            except BaseException:
                pass
    if flashlight.__version__.endswith('dirty'):
        detected_libs.append(('flashlight', f'!!! {flashlight.__version__}  !!!'))
    else:
        detected_libs.append(('flashlight', flashlight.__version__))

    if len(detected_libs):
        detected_libs.sort(key=lambda x: x[0].lower())
        s = tt.to_string(tuple(detected_libs), header=['package name', 'version'])
        logger.info('\n' + indent(f'Imported non-standard libraries:\n{s}', '    '))


def auto_log(cfg):
    date = datetime.now().strftime('%Y_%m%d_%H:%M:%S')
    prefix = cfg.generic.get('log_dir_prefix', '')
    basedir = cfg.generic.get('log_dir', '')
    postfix = cfg.git_hash

    logdir = Path(f'{basedir}/{prefix}{date}_{postfix}')
    logdir.mkdir(parents=True, exist_ok=True)
    log(cfg, logdir)
    return logdir


def auto_init(section=None):
    global engine
    cfg = auto_cfg()

    logdir = auto_log(cfg)

    with open(logdir / 'runtime_config.yaml', 'wt') as f:
        f.write(cfg.yaml)

    logger.success('\n' + indent(pyfiglet.figlet_format('Flashlight', 'cybermedium'), '    '))

    generic_report()

    logger.debug('Configuration:\n' + cfg.yaml)

    model = get_model(cfg.model)
    optimizer = get_optimizer(cfg.model.optimizer)(
        model.parameters(), **cfg.model.optimizer.kwargs)
    scheduler = get_scheduler(cfg.model.scheduler)
    scheduler = scheduler(optimizer, **cfg.model.scheduler.kwargs)
    loss = LossEvaluator(cfg.loss)
    data_loaders = get_dataloaders(cfg.data)

    if section is None:
        engine = Engine(cfg, model, loss, optimizer, scheduler, data_loaders)
    else:
        engine = Engine(cfg[section], model, optimizer, scheduler, data_loaders)

    summary_writers = SBox(default_box=True)
    for phase in cfg.data:
        for name in cfg.data[phase]:
            summary_writers[phase][name] = SummaryWriter(logdir=logdir / f'{phase}-{name}', filename_suffix=f'_{phase}-{name}', flush_secs=5)
    engine.state.summary_writers = summary_writers
    engine.state.logdir = logdir
    engine.enable_automagic()
    return engine


def auto_cfg():
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        filenames = ('config.yaml', 'local_config.yaml')
    cfg = cfg_reader(filenames)
    cfg = resolve_environment(cfg)
    cfg = ensure_reproducible_results(cfg)

    return cfg

__all__ = ['auto_cfg', 'auto_init', 'auto_log', 'generic_report']
