#!/usr/bin/env python3

# Flashlight

__version__ = '0.1'
__description__ = 'Flashlight: rapid prototyping library for Pytorch'
__summary__ = 'Flashlight: rapid prototyping library for Pytorch'
__license__ = 'BSD'
__author__ = 'David Völgyes'
__email__ = 'david.volgyes@ieee.org'

from loguru import logger
from flashlight.auto import auto_init
import flashlight.env
import flashlight.engine
import flashlight.auto
import flashlight.credits
import os
from dotenv import load_dotenv
load_dotenv()
