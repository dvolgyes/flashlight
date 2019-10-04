#!/usr/bin/env python3
import torch
from pathlib import Path
import nibabel
import numpy as np
from loguru import logger
from box import Box


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

    def initialize(self):
        for key in self.items:
            nib = nibabel.load(str(self.items[key]))
            enc = self.config.encodings[key]

            dtype = None
            if enc == 'one_hot':
                dtype = np.dtype('uint8')
            else:
                dtype = np.dtype(enc)

            if dtype.kind == 'f':
                vol = nib.get_fdata(dtype=dtype)
                indices = np.pad(range(vol.shape[-1]), self.context, mode='reflect')
                self.volumes[key] = torch.from_numpy(np.take(vol, indices, axis=-1))

            else:
                vol = nib.get_data().astype(dtype)
                self.weak_supervision = np.any(vol == self.undefined)
                self.size = vol.shape[-1]
                indices = np.pad(range(vol.shape[-1]), self.context_out, mode='reflect')
                self.volumes[key] = torch.from_numpy(np.take(vol, indices, axis=-1))


            axes = np.roll(range(vol.ndim), vol.ndim - 2)
            self.volumes[key] = self.volumes[key].permute(tuple(axes))

        if self.weak_supervision:
            self.volumes['mask'] = self.volumes['label'] == self.undefined
        else:
            weights = []
            for i in range(self.config.n_classes.label):
                weights.append( float(torch.sum(self.volumes['label']==i)) )
            weights = np.asarray(weights)
            weights = 1.0 / weights
            weights = weights / weights.sum()
            self.volumes['mask'] = torch.Tensor().new_ones(self.volumes['label'].shape, dtype=bool)
            self.volumes['class_weights'] = torch.from_numpy(weights.astype(np.float32))
            logger.debug(f'Weights for volume "{self.items[key]}": {np.round(weights,4)}')

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
                result[key] = self.volumes[key][idx:idx + self.out_window, ...].unsqueeze(0)
            else:
                result[key] = self.volumes[key]
                # ~ raise ValueError('unexpected label')
        return Box(result)

    def __len__(self):
        if len(self.volumes) == 0:
            self.initialize()
        return self.size


#~ class PatientDB(torch.utils.data.Dataset):

    #~ def __init__(self, config):
        #~ config.generic.directory = Path(config.generic.directory)
        #~ self.config = config
        #~ self.volumes = []
        #~ self.superbatch = 1
        #~ for item in self.config.data:
            #~ for k in item:
                #~ if isinstance(item[k], str):
                    #~ item[k] = config.generic.directory / item[k]
            #~ self.volumes.append(MedVolume(item, config.generic))

    #~ def __len__(self):
        #~ return len(self.volumes)

    #~ def __getitem__(self, idx):
        #~ if idx < 0 or idx >= len(self):
            #~ raise IndexError

        #~ return self.volumes[idx]

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
            for i in v:
                self.mapping[idx] = (v_idx, i)
                idx += 1
        self.size = idx

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError

        v_idx, s_idx = self.mapping[idx]
        return self.volume[v_idx][s_idx]
