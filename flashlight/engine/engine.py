#!/usr/bin/env python3
from box import SBox
from functools import partial
from loguru import logger
import torch
import warm
import numpy as np
from flashlight.util import dcn

DBox = partial(SBox, default_box=True)


class Engine:

    def __init__(self, cfg, model, loss, optimizer, scheduler, dataloaders, **kwargs):
        self.state = DBox()
        self.cfg = cfg
        self.handlers = {}
        self.state.terminate = False
        self.model = model
        self.loss_function = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders

        logger.info('Model description:\n'+warm.util.summary_str(self.model))

    def _fire(self, event, log_remark=''):
        if event not in self.handlers:
            self.handlers[event] = []
        logger.warning(f'{log_remark}{event}')
        for handler in self.handlers[event]:
            handler(self.state)

    def add_event_handler(self, event, func):
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(func)

    def remove_event_handler(self, event, func):
        try:
            self.handlers[event].remove(func)
        except ValueError:
            pass

    def on(self, event, *args, **kwargs):
        def decorator(f):
            # ~ logger.warning(f"Handler {f.__name__}() from {f.__code__.co_filename}:{f.__code__.co_firstlineno} is registered for event '{event}'.")
            self.add_event_handler(event, f, *args, **kwargs)
            return f
        return decorator

    def run(self, *args, **kwargs):

        self.state.epoch = 0
        self.state.iteration = 0

        self.optimizer.zero_grad()

        self._fire('STARTED')
        self.cfg.engine.super_batch = self.cfg.engine.get('super_batch', '1')
        while self.state.epoch < self.cfg.engine.num_epochs or self.state.terminate:  # EPOCH
            self.model.train()
            self.state.epoch += 1
            self._fire('EPOCH_STARTED', f'Epoch #{self.state.epoch}: ')
            self.state.super_batch = 0
            self.state.optimizer_steps = 0
            self.state.iteration += 1
            for dataloader_name, dataloader in self.dataloaders['train'].items():
              if hasattr(dataloader,'super_batch'):
                super_batch = dataloader.super_batch
              else:
                super_batch = 1

              for b, dataset in enumerate(dataloader):    # ITERATION
                self._fire('ITERATION_STARTED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}: ')
                if not isinstance(dataset, torch.utils.data.Dataset):
                    dataset = (dataset,)

                loss = 0
                for k,data  in enumerate(dataset):
                    self._fire('BATCH_LOADED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch} mini-batch #{k}: ')
                    self._fire('BATCH_FINISHED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch} mini-batch #{k}: ')

                    output = self.model(data['input'])


                    total, lossDict = self.loss_function(output, data)
                    loss = dcn(total) + dcn(loss)
                    total.backward()
                    self.state.loss = dcnm(lossDict)

                    self._fire('LOSS_UPDATED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch}: ')

                if self.state.super_batch+1  >= super_batch:
                    self._fire('OPTIMIZATION', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch}: ')
                    self.optimizer.step()
                    self.state.optimizer_steps += 1
                    self._fire('OPTIMIZATION_FINISHED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch}: ')
                    self.state.super_batch = 0
                    if 'metrics' in self.scheduler.step.__code__.co_varnames:
                        self.scheduler.step(metrics=loss)
                    else:
                        self.scheduler.step()
                    self._fire('SCHEDULER_FINISHED', f'Optimization step #{self.state.optimizer_steps}, learning rate(s): ')
                    self.optimizer.zero_grad()
                    self._fire('ITERATION_FINISHED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}: ')
                    self.state.iteration += 1

                self.state.super_batch += 1

              if self.state.super_batch > 1:
                self._fire('OPTIMIZATION', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch}: ')

                self.optimizer.step()
                self.state.optimizer_steps += 1
                self._fire('OPTIMIZATION_FINISHED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.batch}: ')
                self.state.super_batch = 0
                self.scheduler.step()
                self._fire('SCHEDULER_FINISHED', f'Optimization step #{self.state.optimizer_steps}, learning rate(s): ')
                self.optimizer.zero_grad()
                self._fire('ITERATION_FINISHED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}: ')
                self.state.iteration += 1
            sys.exit(0)
            # ~ self.model.eval()
            # ~ self._fire('PRE_VALIDATION', f'Epoch #{self.state.epoch}:')

            # ~ for dataloader in self.dataloaders['train']:
              # ~ for b, dataset in enumerate(dataloader): # ITERATION

                # ~ self._fire('VALIDATION_ITERATION_STARTED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}: ')

                # ~ if not isinstance(dataset, torch.utils.data.Dataset):
                    # ~ dataset = (dataset,)
                # ~ for data in dataset:
                    # ~ pass

                    # ~ loss = loss + loss_fn(output)
                    # model, loss
                    # ~ self._fire('LOSS_UPDATED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch}: ')

            # ~ self._fire('VALIDATION_FINISHED', f'Epoch #{self.state.epoch}: ')
            # ~ self._fire('EPOCH_FINISHED', f'Epoch #{self.state.epoch}: ')
