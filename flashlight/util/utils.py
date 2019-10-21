#!/usr/bin/env python3
from decopatch import function_decorator, DECORATED
from makefun import wraps
from gc import collect
import numpy as np
from box import SBox, Box
from tensorboardX import SummaryWriter

from datetime import datetime
from functools import partial
from importlib import import_module
from loguru import logger
from pathlib import Path
from contracts import contract
import operator


@function_decorator
def gc(f=DECORATED):
    """
    This decorator calls garbage collection before & after the function.
    """

    @wraps(f)
    def new_f(*args, **kwargs):
        collect()
        result = f(*args, **kwargs)
        collect()
        return result

    return new_f


@function_decorator
def gc_before(f=DECORATED):
    """
    This decorator calls garbage collection before the function.
    """

    @wraps(f)
    def new_f(*args, **kwargs):
        collect()
        result = f(*args, **kwargs)
        return result

    return new_f


@function_decorator
def gc_after(f=DECORATED):
    """
    This decorator calls garbage collection after the function.
    """

    @wraps(f)
    def new_f(*args, **kwargs):
        result = f(*args, **kwargs)
        collect()
        return result

    return new_f


def dict_op(d1, d2, op=operator.add):
    result = SBox(default_box=True)
    for key in set(d1.keys()) & set(d2.keys()):
        result[key] = op(d1[key], d2[key])
    for key in set(d1.keys()) - set(d2.keys()):
        result[key] = d1[key]
    for key in set(d2.keys()) - set(d1.keys()):
        result[key] = d2[key]
    return result


def dcns(*args):
    results = []
    for x in args:
        if isinstance(x, (dict, SBox, Box)):
            for key in x.keys():
                x[key] = dcns(x[key])
        if hasattr(x,'detach') and callable(getattr(x,'detach')):
            x = x.detach().cpu().numpy()
        if isinstance(x,np.ndarray) and x.size==1:
            x = np.sum(x)
        results.append(x)
    if len(results) == 1:
        return results[0]
    return tuple(results)

def dcn(*args):
    results = []
    for x in args:
        if isinstance(x, (dict, SBox, Box)):
            for key in x.keys():
                x[key] = dcn(x[key])
        if not isinstance(x, (int, float, np.uint8, np.uint16, np.uint32, np.uint64, bool, np.bool, np.bool_,
                              np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.ndarray, dict, SBox, Box)):
            results.append(x.detach().cpu().numpy())
        else:
            results.append(x)
    if len(results) == 1:
        return results[0]
    return tuple(results)


def dcnm(*args):
    results = []
    for arg in args:
        d = dcn(arg)
        if isinstance(d, np.ndarray):
            results = np.mean(dcn(arg))
        else:
            results = d

    if len(results) == 1:
        return results[0]
    return tuple(results)


@contract(name='str')
def import_function(name):
    relatives = 0
    base = name.strip('.')
    relatives = len(name) - len(base)
    module, *parts = base.split('.')
    module = ('.' * relatives) + module
    result = import_module(module)
    for part in parts:
        if hasattr(result, part):
            result = getattr(result, part)
        else:
            module = f'{module}.{part}'
            result = import_module(module)
    return result


def avg_dict(a, b):
    result = SBox(default_box=True)
    for key in set(a.keys()) | set(b.keys()):
        result[key] = (a.get(key, 0) + b.get(key, 0)) / 2.0
    return result


class SWDispatcher:

    def __init__(self, writers):
        self.writers = writers

    def __getattr__(self, name):
        if not name.startswith('_'):
            return partial(self.dispatch, name)
        raise AttributeError()

    def close(self):
        for k in self.writers:
            self.writers[k].close()

    def _add_figure(self, writer_key, key, fig, epoch):
        writer = self.writers[writer_key]
        filename = Path(writer.logdir) / f'steps/{epoch:06d}' / (key + '.png')
        filename.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        fig.savefig(filename)
        logger.error(f'{filename} was created.')

    def dispatch(self, func, key, *args, **kwargs):
        f = getattr(self.writers[key], func)
        f(*args, **kwargs)

        f = getattr(self.writers['combined'], func)
        suffixed = f'{args[0]}_{key}'
        f(suffixed, *args[1:], **kwargs)

        if hasattr(self, f'_{func}'):
            f = getattr(self, f'_{func}')
            f(key, *args, **kwargs)
            suffixed = f'{args[0]}_{key}'
            f('combined', suffixed, *args[1:], **kwargs)


def get_summary_writers(cfg):
    writers = SBox(default_box=True)
    date = datetime.now().strftime('%Y_%m%d_%H:%M:%S')
    basedir = cfg.runtime.tensorboard_dirs.separate
    prefix = cfg.runtime.get('log_dir_prefix', '')
    postfix = cfg.git_hash
    for phase in cfg.runtime.phases:
        logdir = f'{basedir}/{prefix}{date}_{phase}_{postfix}'
        writers[phase] = SummaryWriter(logdir=logdir)
    basedir = cfg.runtime.tensorboard_dirs.combined
    logdir = f'{basedir}/{prefix}{date}_{postfix}'
    writers['combined'] = SummaryWriter(logdir=logdir)

    return SWDispatcher(writers), logdir
