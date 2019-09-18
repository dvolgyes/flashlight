#!/usr/bin/env python3
import yaml
from box import SBox
from pathlib import Path
from flashlight.util import resolve_templates


def cfg_reader(filenames):
    def _read(fname):
        with open(fname, 'r') as stream:
            try:
                cfg = SBox(yaml.safe_load(stream), default_box=True)
            except ValueError:
                cfg = SBox(default_box=True)
        return cfg

    cfg = SBox(default_box=True)
    for fname in filenames:
        if Path(fname).exists():
            cfg.update(_read(fname))

    return resolve_templates(cfg)
