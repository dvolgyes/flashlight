#!/usr/bin/env python3
import torch
from pathlib import Path
import nibabel
import numpy as np


def one_hot_encoding(array, classes, dtype=np.uint8, swapaxes=False):  # for single batch
    if classes < np.unique(array).size:
        raise ValueError('One hot encoding Number of classes is less than the number of unique values.')
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
        self.window = 1 + 2 * self.context
        if config.data_type not in ('nibabel'):
            raise ValueError('Only nibabel format is accepted.')
        self.volumes = {}
        self.weak_supervision = False
        self.superbatch = 1

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
                indices = np.pad(range(vol.shape[-1]), context, mode='reflect')
                self.volumes[key] = torch.from_numpy(np.take(vol, indices, axis=-1))
            else:
                vol = nib.get_data().astype(dtype)
                self.size = vol.shape[-1]
                self.weak_supervision = np.any(vol == self.undefined)
                self.volumes[key] = torch.from_numpy(vol)

            axes = np.roll(range(vol.ndim), vol.ndim - 2)
            self.volumes[key] = self.volumes[key].permute(tuple(axes))

        if self.weak_supervision:
            self.volumes['mask'] = self.volumes['label'] == self.undefined
        else:
            self.volumes['mask'] = torch.Tensor().new_ones(self.volumes['label'].shape, dtype=bool)

    def __getitem__(self, idx):
        if len(self.volumes) == 0:
            self.initialize()

        if idx < 0 or idx >= len(self):
            raise IndexError

        result = {'weight': self.weight,
                  'weak_supervision': self.weak_supervision}

        for key in self.volumes:
            if key == 'input':
                result[key] = self.volumes[key][idx:idx + self.window, ...].unsqueeze(0)
            elif key in ['label', 'mask']:
                result[key] = self.volumes[key][idx, ...].unsqueeze(0)
            else:
                pass
                # ~ raise ValueError('unexpected label')
        return result

    def __len__(self):
        if len(self.volumes) == 0:
            self.initialize()
        return self.size


class PatientDB(torch.utils.data.Dataset):

    def __init__(self, config):
        config.generic.directory = Path(config.generic.directory)
        self.config = config
        self.volumes = []
        self.superbatch = 1
        for item in self.config.data:
            for k in item:
                if isinstance(item[k], str):
                    item[k] = config.generic.directory / item[k]
            self.volumes.append(MedVolume(item, config.generic))

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError

        return self.volumes[idx]
