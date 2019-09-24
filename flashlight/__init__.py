#!/usr/bin/env python3

# Flashlight

__version__ = '0.1'
__description__ = 'Flashlight: rapid prototyping library for Pytorch'
__summary__ = 'Flashlight: rapid prototyping library for Pytorch'
__license__ = 'BSD'
__author__ = 'David Völgyes'
__email__ = 'david.volgyes@ieee.org'

import flashlight.util  # noqa: E402, F401
from flashlight.auto import auto_init
from dotenv import load_dotenv
load_dotenv()

import flashlight.auto  # noqa: E402
import flashlight.credits  # noqa: E402
import flashlight.engine  # noqa: E402
import flashlight.env  # noqa: E402
import flashlight.loss  # noqa: E402
import flashlight.models  # noqa: E402
import flashlight.optim  # noqa: E402
import flashlight.plot  # noqa: E402
import flashlight.data  # noqa: E402, F401


__all__ = ['auto_init']
