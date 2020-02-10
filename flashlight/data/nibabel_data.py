#!/usr/bin/env python3
import torch
from pathlib import Path
import nibabel
import numpy as np
from loguru import logger
from box import SBox
import matplotlib.pyplot as plt


def str_to_dtype(s):
    if s in ['float16', 'half']:
        return torch.float16
    elif s in ['float32', 'single', 'float']:
        return torch.float32
    elif s in ['float64', 'double']:
        return torch.float64
    elif s in ['int8']:
        torch.int8
    elif s in ['uint8']:
        torch.uint8
    elif s in ['int16', 'short']:
        torch.int16
    elif s in ['int32', 'int']:
        torch.int32
    elif s in ['int64', 'long']:
        torch.int64
    else:
        raise ValueError(f'invalid dtype: {s}')


class MedVolume(torch.utils.data.Dataset):

    def __init__(self, items, config):
        self.weight = items.pop('weight', 1.0)
        self.undefined = items.pop('undefined_class', -1)
        self.items = items
        self.config = config
        self.context = config.context
        self.window = 1 + 2 * self.context
        self.out_window = self.config.output_layers
        self.context_out = self.out_window - self.out_window // 2

        if config.data_type not in ('nibabel'):
            raise ValueError('Only nibabel format is accepted.')
        self.volumes = {}
        self.weak_supervision = False
        self.superbatch = 1

        nib = nibabel.load(str(self.items['label']))
        self.size = nib.shape[-1]
        logger.trace(f'Nibabel label: {self.items["label"]}, shape: {nib.shape}')

    def initialize(self):
        for key in self.items:
            self.config.encodings[key].public = self.config.encodings[key].get('public', 'float32')
            self.config.encodings[key].internal = self.config.encodings[key].get('internal', 'float16')

            nib = nibabel.load(str(self.items[key]))
            enc = self.config.encodings[key].public
            dtype = np.dtype(enc)

            if dtype.kind == 'f':
                vol = nib.get_fdata(dtype=dtype)
                A = vol[..., 1:self.context + 1][..., ::-1]
                B = vol[..., -self.context - 1:-1][..., ::-1]
                self.volumes[key] = torch.from_numpy(np.clip((np.dstack((A, vol, B)) - 100.) / 128., -0.7, 0.7))

                #~ if self.config.internal_representation in ['half', 'float16']:
                    #~ self.volumes[key] = self.volumes[key].half()
                    #~ logger.info('Float32 to float16 conversion.')
            else:
                vol = nib.get_data().astype(dtype)
                self.weak_supervision = np.any(vol == self.undefined)
                self.size = vol.shape[-1]
                A = vol[..., 1:self.context + 1][..., ::-1]
                B = vol[..., -self.context - 1:-1][..., ::-1]
                self.volumes[key] = torch.from_numpy(np.dstack((A, vol, B)))

            axes = np.roll(range(vol.ndim), vol.ndim - 2)
            internal_dtype = str_to_dtype(self.config.encodings[key].get('internal'))
            self.volumes[key] = self.volumes[key].permute(tuple(axes)).to(internal_dtype)

        if self.weak_supervision:
            self.volumes['mask'] = self.volumes['label'] == self.undefined
        else:
            weights = []
            for i in range(self.config.n_classes.label):
                weights.append(float(torch.sum(self.volumes['label'] == i)))
            weights = np.asarray(weights, dtype=np.float32)
            w_mask = np.sign(weights) # if weight was 0, keep it zero
            weights = w_mask * (weights.size / (np.asarray(weights)))
            weights = weights / (weights.sum() + 1e-3)

            self.volumes['mask'] = torch.Tensor().new_ones(self.volumes['label'].shape, dtype=bool)
            self.volumes['class_weights'] = torch.from_numpy(weights.astype(np.float32))
            logger.trace(f'Weights for volume "{self.items[key]}": {weights}')

    def __getitem__(self, idx):
        if len(self.volumes) == 0:
            self.initialize()

        if idx < 0 or idx >= len(self):
            raise IndexError

        result = {'weight': self.weight,
                  'weak_supervision': self.weak_supervision}

        for key in self.volumes:
            if key == 'input':
                result[key] = self.volumes[key][idx:idx + self.window, ...]
            elif key in ['label', 'mask']:
                result[key] = self.volumes[key][idx:idx + self.out_window, ...]
            else:
                result[key] = self.volumes[key]
                # ~ raise ValueError('unexpected label')
            if hasattr(result[key], 'dtype') and 'internal' in self.config.encodings[key]:
                dtype = str_to_dtype(self.config.encodings[key].public)
                result[key] = result[key].to(dtype)

        return SBox(result, default_box=True)

    def __len__(self):
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
        self.mapping = {}
        idx = 0
        for v_idx, v in enumerate(self.volumes):
            for i in range(len(v)):
                self.mapping[idx] = (v_idx, i)
                idx += 1
        self.size = idx

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError

        v_idx, s_idx = self.mapping[idx]
        result = self.volumes[v_idx][s_idx]

        return result
