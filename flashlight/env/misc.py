#!/usr/bin/env python3
import sys
import os
from contextlib import contextmanager
from pathlib import Path
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters.terminal import TerminalFormatter
import pkg_resources
import getversion
from termcolor import colored


def getattrs(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


@contextmanager
def local_env(name, value):
    defined = name in os.environ
    if defined:
        saved = os.environ[name]
    os.environ[name] = str(value)

    yield

    if defined:
        os.environ[name] = saved
    else:
        os.environ.pop(name)


def env_variable(name, global_env):
    if global_env in os.environ:
        return type(name)(os.environ[global_env])
    return name


def detect_libs(lib, name=None):
    enabled, version = False, 'N/A'
    enabled = lib in sys.modules

    if enabled:
        mod = sys.modules[lib]
    else:
        mod = importlib.import_module(lib)

    version = getversion.get_module_version(mod)[0]

    if name is None:
        try:
            name = pkg_resources.get_distribution(lib).project_name
        except:
            name = lib

    return {
        'version': version,
        'enabled': enabled,
        'name': name
    }


def free_space(dirname='.'):
    try:
        f = os.statvfs(dirname)
        return (f.f_bavail * f.f_bsize / (1024**3))
    except Exception:
        return -1


def find_global_variable_name(var):
    for k, v in globals().items():
        if v is var:
            return k
    return None


def path_difference(file1, file2):
    diff = str(Path(os.path.relpath(file1, file2)))
    cnt = 0
    for c in diff:
        if c not in ['.', '/']:
            break
        if c == '/':
            cnt += 1
    return cnt


def format_trace(trace, skip=0, levels=None):
    global _settings
    lines = []
    f0 = Path(trace[0].filename)
    for idx, t in enumerate(trace):
        if idx < skip:
            continue
        if levels is not None and idx + levels >= len(trace):
            break
        e = Path(t.filename).name + ':' + str(t.lineno)
        if path_difference(f0, Path(t.filename)) <= 3:
            e = colored(e, 'green')
        b = '  ' * idx + t.line
        n = max(_settings.line_length - len(b), 0)
        b = '  ' * idx + (highlight(t.line, PythonLexer(),
                                    TerminalFormatter())).strip()
        e = e.strip()
        lines.append(b + (' ' * n) + e)
    return '\n'.join(lines)
