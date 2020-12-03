#!/bin/bash
python3 hyper.py hyper.pcs >hyper.yaml
python3 example.py config.yaml local.yaml hyper.yaml
