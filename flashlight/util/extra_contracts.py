#!/usr/bin/env python3
from contracts import contract, new_contract
from pathlib import Path

__placeholder__ = 'placeholder'


@new_contract
def path(p):
    """New contract type: path which is str or Path."""
    return isinstance(p, (Path, str))


@new_contract
@contract(p='path')
def existing_file(p):
    """Check if the variable points to existing file."""
    path = Path(p)
    return path.exists() and path.is_file()


@new_contract
@contract(p='path')
def existing_dir(p):
    """Check if the variable points to existing directory."""
    path = Path(p)
    return path.exists() and path.is_dir()


@new_contract
@contract(p='path')
def new_file(p):
    """Check if the variable points to a non-existing file."""
    path = Path(p)
    return path.parent.exists() and not path.exists()


@new_contract
@contract(p='path')
def new_dir(p):
    """Check if the variable points to a non-existing directory."""
    path = Path(p)
    return path.parent.exists() and not path.exists()
