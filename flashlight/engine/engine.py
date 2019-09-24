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
        logger.info('Model description:\n' + warm.util.summary_str(self.model))
        self.state.summary_writers = DBox()

    def enable_automagic(self):
        self.add_event_handler('LOSS_UPDATED', self.log_loss)
        self.add_event_handler('CHECKPOINT', self.checkpoint_model)

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

    def log_loss(self, state):
        for loss_name in state.loss:
            value = dcn(state.loss)[loss_name]
            state.summary_writers[state.phase][state.sub_phase].add_scalar(f'loss/{loss_name}', value, state.iteration)
            logger.success(f'loss/{loss_name}: {value}')

    def checkpoint_model(self, state):
        torch.save(self.model.state_dict(), state.logdir / f'checkpoint_model_{state.iteration}.pth')
        torch.save(self.optimizer.state_dict(), state.logdir / f'checkpoint_optim_{state.iteration}.state')

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
                self.state.phase = 'train'
                self.state.sub_phase = dataloader_name
                self.state.diversity = 0
                self.state.diversity_data = None
                self.state.diversity_out = None

                if hasattr(dataloader, 'super_batch'):
                    super_batch = dataloader.super_batch
                else:
                    super_batch = 1

                for dataset in dataloader:    # ITERATION
                    self._fire('ITERATION_STARTED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}: ')
                    if not isinstance(dataset, torch.utils.data.Dataset):
                        dataset = (dataset,)

                    loss = 0
                    for k, data in enumerate(dataset):
                        self._fire(
                            'BATCH_LOADED',
                            f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch} mini-batch #{k}: ')
                        self._fire('BATCH_FINISHED',
                                   f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch} mini-batch #{k}: ')

                        output = self.model(data['input'])

                        total, lossDict = self.loss_function(output, data)

                        loss = dcn(total) + dcn(loss)
                        total.backward()

                        self.state.loss = dcn(lossDict)

                        self.state.data = dcn(data)
                        self.state.output = dcn(output)

                        diversity = np.sum(self.state.data['label'] == 2)
                        if diversity > self.state.diversity or self.state.diversity_data is None:
                            self.state.diversity = diversity
                            self.state.diversity_data = self.state.data
                            self.state.diversity_out = self.state.output

                        self._fire('LOSS_UPDATED', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.super_batch}: ')

                    if self.state.super_batch + 1 >= super_batch:
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
                        self._fire('CHECKPOINT')
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

            # ~ self.model.eval()
            # ~ self._fire('PRE_VALIDATION', f'Epoch #{self.state.epoch}:')

            # ~ for dataloader in self.dataloaders['validation']:
                # ~ for b, dataset in enumerate(dataloader): # ITERATION
                # ~ self.state.phase='validation'
                # ~ self.state.sub_phase = dataloader_name
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
