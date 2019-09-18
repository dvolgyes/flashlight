#!/usr/bin/env python3
import sys
import os
from contextlib import contextmanager
import importlib
from pathlib import Path
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters.terminal import TerminalFormatter


def colored(x, y):
    return x


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
    enabled, exists, version = False, False, 'N/A'
    try:
        enabled = lib in sys.modules
        importlib.import_module(lib)
        exists = True
        module = sys.modules[lib]
        version = getattrs(
            module, ('__version__', 'VERSION', 'version'), 'unknown')
        if version == 'unknown':
            try:
                v = importlib.import_module(f'{lib}.version')
                version = getattrs(
                    v, ('__version__', 'VERSION', 'version'), 'unknown')
            except ImportError:
                pass
        if isinstance(version, tuple):
            version = '.'.join(map(str, version))
        version = str(version)
        if name is None:
            name = getattrs(module, ('__name__',), lib)
    except ImportError:
        pass
    if name is None:
        name = lib
    if not exists:
        version = 'not installed'
    return {
        'exists': exists,
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

    # ~ self.default_libs = {
    # ~ 'torch': 'PyTorch',
    # ~ 'tensorflow': 'TensorFlow',
    # ~ 'tensorboard': 'TensorBoard',
    # ~ 'tensorboardX': 'TensorBoardX',
    # ~ 'numpy': 'NumPy',
    # ~ 'scipy': 'SciPy',
    # ~ }
    # ~ self.TF_CUDA = False
    # ~ self.CUDA_version = 'unknown'
    # ~ self.CONDA = 'not in use'
    # ~ self.detected_libs = {}
    # ~ self.detect_enviroment()
    # ~ self.detect_default_libs()
    # ~ self.detect_cuda()
    # ~ self.detect_extra_libs()

    # ~ def __getitem__(self, name):
    # ~ if name in self.detected_libs:
    # ~ return self.detected_libs[name]
    # ~ return None

    # ~ def detect_default_libs(self):
    # ~ self.detected_libs['vivisection'] = {
    # ~ 'exists': True,
    # ~ 'version': __version__,
    # ~ 'enabled': True,
    # ~ 'name': 'Vivisection'
    # ~ }
    # ~ for lib, name in self.default_libs.items():
    # ~ if lib not in self.detected_libs:
    # ~ self.detected_libs[lib] = detect_libs(lib, name)

    # ~ def detect_enviroment(self):
    # ~ self.CONDA = os.environ.get('CONDA_DEFAULT_ENV', 'Not in use')

    # ~ def set_logger(self, logger):
    # ~ self.sample_logger = logger

    # ~ def detect_extra_libs(self):
    # ~ self.requirements = Path(self.requirements)
    # ~ if self.requirements.exists():
    # ~ with open(self.requirements, 'rt') as f:
    # ~ for line in f:
    # ~ lib = line.strip().split()
    # ~ if len(lib)>0:
    # ~ lib = lib[0]
    # ~ if lib in sys.modules:
    # ~ self.detected_libs[lib] = detect_libs(lib)

    # ~ def detect_cuda(self):
    # ~ if env_variable(True, 'VIVISECTION_DETECT_CUDA'):
    # ~ if self['tensorflow']['exists']:
    # ~ with local_env('TF_CPP_MIN_LOG_LEVEL', 2):
    # ~ import tensorflow as tf
    # ~ self.TF_CUDA = tf.test.is_gpu_available(cuda_only='True')
    # ~ self['tensorflow']['name'] += "_CUDA"
    # ~ if self['torch']['exists']:
    # ~ self.CUDA_version = torch.version.cuda

    # ~ def log(self):
    # ~ logger.info('System information')
    # ~ logger.info(
    # ~ f' Python          :    {sys.version[0:5]},'
    # ~ f'   Conda:{self.CONDA}, CUDA: {self.CUDA_version}')

    # ~ for key in self.detected_libs.keys():
    # ~ lib = self.detected_libs[key]
    # ~ if lib['exists']:
    # ~ logger.info(
    # ~ f' {lib["name"]:15s} :    {lib["version"]:8s}'
    # ~ f' (imported: {lib["enabled"]})')

    # ~ if not self.requirements.exists():
    # ~ logger.warning(f'Missing requirements.txt!')


# ~ _settings = __vivisection_settings()
# ~ _settings.log()
