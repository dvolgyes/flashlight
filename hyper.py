#!/usr/bin/env python3
import os
import sys
from ConfigSpace.read_and_write import pcs_new as pcs
from duecredit import due, BibTeX
from box import SBox
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
for dirname in os.environ.get('EXTRA_PYTHON_LIBS', '.').split(';'):
    sys.path.append(f'{Path(__file__).parent/dirname}/')

pcs.read = due.dcite(
    BibTeX("""@article{BOAH,
title   = {BOAH: A Tool Suite for Multi-Fidelity Bayesian Optimization & Analysis of Hyperparameters},
author  = {M. Lindauer and K. Eggensperger and M. Feurer and A. Biedenkapp and J. Marben and P. Müller and F. Hutter},
journal = {arXiv:1908.06756 {[cs.LG]}},
date    = {2019},}"""
           ),
    description='Hyperparameter optimization',
    path='ConfigSpace',
)(pcs.read)


def _read_without_comments(fname):
    with open(fname) as fh:
        lines = map(str.strip, fh.readlines())
        text = '\n'.join(x for x in lines if not (x.startswith('#') or x.startswith('\n')))
        return text

with open(sys.argv[1]) as fh:
    hyper_space = pcs.read(fh)

config = hyper_space.sample_configuration()
config_dict = config.get_dictionary()
config_box = SBox(default_box=True, dots=True)
for p, v in config_dict.items():
    config_box[f'hyperparameters.{p}'] = v
print(config_box.to_yaml(default_flow_style=False))   # noqa: T001
