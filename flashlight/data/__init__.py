#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import nibabel


def one_hot_encoding(array, classes, dtype=np.uint8, swapaxes=False):  # for single batch
    shape = array.shape + (classes,)
    x = np.zeros(shape, dtype=dtype)
    for i in range(classes):
        x[..., i][array == i] = 1
    if swapaxes:
        n, m = len(x.shape) - 2, len(x.shape) - 1
        x = np.swapaxes(x, n, m)
    return x


class MedVolume(torch.utils.data.Dataset):

    def __init__(self, items, config):
        self.weight = items.pop('weight', 1.0)
        self.undefined = items.pop('undefined_class', -1)
        self.items = items
        self.config = config
        self.context = config.context
        if config.data_type not in ('nibabel'):
            raise ValueError('Only nibabel format is accepted.')
        self.volumes = {}
        self.weak_supervision = False

    def initialize(self):
        for key in self.items:
            nib = nibabel.load(str(self.items[key]))
            enc = self.config.encodings[key]
            context = self.config.context
            dtype = None
            if enc == 'one_hot':
                dtype = np.dtype('uint8')
            else:
                dtype = np.dtype(enc)

            if dtype.kind == 'f':
                vol = nib.get_fdata(dtype=dtype)
            else:
                vol = nib.get_data().astype(dtype)
                if enc == 'one_hot':
                    self.weak_supervision = np.any(vol == self.undefined)
                    vol = one_hot_encoding(
                        vol, self.config.n_classes[key], dtype=dtype, swapaxes=True)

            self.size = vol.shape[-1]
            indices = np.pad(range(vol.shape[-1]), context, mode='reflect')
            self.volumes[key] = np.take(vol, indices, axis=-1)
            self.window = 1 + 2 * self.context

    def __getitem__(self, arg):
        if len(self.volumes) == 0:
            self.initialize()

        result = {'weight': self.weight,
                  'weak_supervision': self.weak_supervision}
        for key in self.volumes:
            result[key] = self.volumes[key][..., arg:arg + self.window]
        return result

    def __len__(self):
        return self.size


class PatientDB(torch.utils.data.Dataset):

    def __init__(self, config):
        config.generic.directory = Path(config.generic.directory)
        self.config = config
        self.volumes = []
        for item in self.config.data:
            for k in item:
                if isinstance(item[k], str):
                    item[k] = config.generic.directory / item[k]
            self.volumes.append(MedVolume(item, config.generic))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.volumes[idx]
