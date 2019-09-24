#!/usr/bin/env python3
import torch
from box import SBox
from flashlight.util import box_from_file
from .nibabel_data import PatientDB, MedVolume


def get_dataloaders(cfg):
    dataloaders = SBox(default_box=True)
    for phase in cfg:
      for key, data in cfg[phase].items():
        if data.type == 'PatientDB':
            dataset = PatientDB(box_from_file(data.source))
        if 'sampler' in data:
            sampler = getattr(torch.utils.data,data.sampler.name)(dataset, **data.sampler.kwargs)
            data.dataloader.sampler = sampler
        else:
            sampler = None
        print(dataset[0][0])
        dl = torch.utils.data.DataLoader(dataset, **data.dataloader.kwargs)
        dl.super_batch = data.dataloader.get('super_batch', 1)
        dataloaders[phase][key] = dl
    return dataloaders
