#!/usr/bin/env python3
from contracts import contract
from box import SBox
import dpath
import os
import yaml
import torch
import numpy as np
import psutil


@contract
def leaf_values(d, sep='/'):
    """Returning full access path iterator for dictionary/box leaf values.
       E.g. for {'a': {'b': 3}} it would yield ('/a/b', 3)

       Parameters:
       :type d: dict
       :type sep: str
       :rtype: Iterable

       >>> list(leaf_values( {'a': {'b': 3}}))
       [('/a/b', 3)]
    """

    for key in d.keys():
        if isinstance(d[key], dict):
            yield from leaf_values(d[key], sep=f'{sep}{key}/')
        else:
            yield f'{sep}{key}', d[key]


@contract
def resolve_environment(cfg, prefix='FLASHLIGHT'):
    """Environment variables starting with {prefix} will be inserted into the dictionary/box.
       E.g. 'FLASHLIGHT_generic_key = value' will be inserted as
             cfg['generic']['key'] = value

       Parameters:
       :type cfg: dict
       :type prefix: str
       :rtype: dict
    """

    for key in os.environ:
        if not key.startswith(prefix):
            continue
        value = yaml.safe_load(f'key: {os.environ[key]}')['key']
        path = key.split('_', 1)[1]
        dpath.util.new(cfg, path, value, separator='_')
    return cfg


@contract
def resolve_templates(cfg, namespace=None):
    """Template variables using %VAR% format will be resolved in the dictionary.
       The variable should refer to a path in the dictionary, with the dot notation.

       Namespace can be used to provide environmental variables, which might be resolved
       on reference, but in general should not be merged directly into 'cfg'.
       For instance:

       >>> resolve_templates({'a': '%X%', 'b': {'c':'d'}, 'e': '%b.c%'}, {'X': 22, 'OS': 'L'})
       {'a': 22, 'b': {'c': 'd'}, 'e': 'd'}

       Missing templates should be unchanged:
       >>> resolve_templates({'a': '%MISSING%'})
       {'a': '%MISSING%'}

       Built-in variables:
       >>> resolve_templates({'a': '%PHYSICAL_CORES%'})
       {'a': 2}

       Built-in variables:
       >>> resolve_templates({'a': '${3+6}'})
       {'a': 9}


       Parameters:
       :type cfg: dict
       :type namespace: None|dict
       :rtype: dict
    """

    if namespace is None:
        namespace = {}
    namespace = SBox(namespace, default_box=False).copy()

    namespace.LOGICAL_CORES = psutil.cpu_count()
    namespace.PHYSICAL_CORES = psutil.cpu_count(logical=False)
    template = 1
    while template > 0:
        template = 0
        for path, value in leaf_values(cfg):
            if isinstance(value, str) and value.startswith('%') and value.endswith('%'):
                meta_var = '/' + (value[1:-1].replace('.', '/'))
                try:
                    v = dpath.util.get(cfg, meta_var)
                    template += 1
                except KeyError:
                    try:
                        v = dpath.util.get(namespace, meta_var)
                        template += 1
                    except:
                        v = value  # unchanged
                dpath.util.new(cfg, path, v)

    if SBox(cfg, default_box=True).generic.enable_python_codes == True or True:
        template = 1
        while template > 0:
            template = 0
            for path, value in leaf_values(cfg):
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    expr = value[2:-1]
                    value = eval(expr)
                    dpath.util.new(cfg, path, value)

    return cfg


if __name__ == '__main__':
    import doctest
    doctest.testmod()
