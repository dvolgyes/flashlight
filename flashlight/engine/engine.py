#!/usr/bin/env python3
from box import SBox
from functools import partial
from loguru import logger
import torch
import warm
import numpy as np
from flashlight.util import dcn
from flashlight.log import clog

import sys
from collections.abc import Iterable

DBox = partial(SBox, default_box=True)


class Engine:
    def __init__(self, cfg, model, loss, optimizer, scheduler, dataloaders, **kwargs):
        self.state = DBox()
        self.cfg = cfg
        self.handlers = DBox()
        self.state.terminate = False
        self.model = model
        self.loss_function = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        model_summary = warm.util.summary_str(self.model)
        model_param_count = int(model_summary.split('\n')[2].split()[1])
        if model_param_count <= 1e4:
            model_param_count = f'{model_param_count}'
        if 1e3 < model_param_count < 1e5:
            model_param_count = f'{model_param_count/1000:.1f}k'
        else:
            model_param_count = f'{model_param_count/1000000:.2f}M'
        self.log('model', 'DEBUG', f'Model description:\n{model_summary}')
        self.log('model_params', 'INFO', f'Trainable model parameters: {model_param_count}')
        if hasattr(model, 'module'):
            if hasattr(model.module, 'receptive_field'):
                self.log('model_receptive_field', 'INFO', f'Receptive field: {model.module.receptive_field}')

        self.state.summary_writers = DBox()

    def log(self, key, default, message='INFO'):
        clog(self.cfg, key, default, message)

    def enable_automagic(self):
        pass

    def _fire(self, event, log_remark=''):

        for handler in self.handlers[f'BEFORE_{event}']:
            self.log('engine.fire', 'TRACE', f'{log_remark}BEFORE_{event} {handler}:  {handler.__name__} from {handler.__code__.co_filename}:{handler.__code__.co_firstlineno}')
            handler(self.state)

        if len(self.handlers[event]):
            for handler in self.handlers[event]:
                self.log('engine.fire', 'TRACE', f'{log_remark}{event}:  {handler.__name__} from {handler.__code__.co_filename}:{handler.__code__.co_firstlineno}')
                handler(self.state)

        elif hasattr(self, event):
            self.log('engine.fire_default', 'TRACE', f'{log_remark}{event}: Default event handler.')
            getattr(self, event)(self.state)
        else:
            self.log('engine.fire_no_default', 'TRACE', f'{log_remark}{event}: No event handler defined.')

        for handler in self.handlers[f'AFTER_{event}']:
            self.log('engine.fire', 'TRACE', '{log_remark}AFTER_{event}:  {handler.__name__} from {handler.__code__.co_filename}:{handler.__code__.co_firstlineno}')
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
            if isinstance(event, Iterable) and not isinstance(event, str):
                for e in event:
                    self.add_event_handler(e, f, *args, **kwargs)
            else:
                self.add_event_handler(event, f, *args, **kwargs)
            return f
        return decorator

    def log_loss(self, state):
        for loss_name in state.loss:
            value = dcn(state.loss)[loss_name]
            state.summary_writers[state.phase][state.sub_phase].add_scalar(f'loss/{loss_name}', value, state.iteration)
            self.log('loss', 'SUCCESS', f'loss/{loss_name}: {value}')
            #~ state.summary_writers[state.phase][state.sub_phase].flush()

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.state.loss_step = 0
        for k in self.state.loss:
            self.state.loss[k] = 0
            self.state.mean_loss[k] = 0
            self.state.cummulative_loss[k] = 0
        self.state.diversity_data = None

    def clear_state(self):
        self.state.data = None
        self.state.output = None
        self.state.total_loss = 0

    def START(self, state):
        self.clear_state()
        self.state.epoch = 0
        self.state.iteration = 1
        self.state.loss_step = 0
        self.state.optimizer_steps = 0
        self.state.total_loss = 0
        self.state.cfg = self.cfg
        self.device = self.cfg.generic.device
        self.model.to(self.device)
        if not hasattr(self.model, 'recurrent_wrap'):
            setattr(self.model, 'recurrent_wrap', None)
        self.optimizer.zero_grad()
        self.cfg.engine.super_batch = self.cfg.engine.get('super_batch', '1')
        self.cfg.model.recurrent_wrap = self.cfg.model.get('recurrent_wrap', '0')

    def EPOCH_START(self, state):
        self.state.super_batch = 0
        self.log('log_dir', 'INFO', f'Working dir: {state.logdir}, current epoch: #{state.epoch}')

    def TRAINING_START(self, state):
        self.state.phase = 'train'
        self.model.train()
        self.clear_state()

    def ITERATION_START(self, state):
        pass

    def ITERATION_END(self, state):
        self.clear_state()

    def TRAINING_END(self, state):
        self.clear_state()

    def BATCH_LOAD(self, state):
        if self.cfg.model.recurrent_wrap:
            state.state.data['hidden'] = model.get_initial_hidden_state()

        for key in self.state.data:  # transfer to device, if applicable
            if hasattr(self.state.data[key], 'to'):
                self.state.data[key] = self.state.data[key].to(self.device)

    def LOSS_UPDATE(self, state):
        # TODO remove reshaping
        self.state.data['label'] = self.state.data['label'].view(-1, 512, 512)
        self.state.output = self.state.output.view(-1, 3, 512, 512)

        total, self.state.loss = self.loss_function(self.state.output, self.state.data)
        self.state.loss_step += 1
        for k in self.state.loss:
            self.state.cummulative_loss[k] = self.state.cummulative_loss.get(k, 0) + self.state.loss[k]
            self.state.mean_loss[k] = self.state.cummulative_loss[k] / self.state.loss_step

        self.state.total_loss = total + self.state.total_loss
        self.state.loss = dcn(self.state.loss)
        self.state.data = dcn(self.state.data)
        self.state.output = dcn(self.state.output)

    def BACKWARD(self, state):
        self.state.total_loss.backward()
        self.state.total_loss = 0 #self.state.loss['total']

    def OPTIMIZATION(self, state):
        if self.state.loss_step > 0:
            self.optimizer.step()
            self.state.optimizer_steps += 1
            self._fire('SCHEDULER')
            self.zero_grad()
            self._fire('ITERATION_END')
            self.state.iteration += 1

    def SCHEDULER(self, state):
        if self.scheduler is not None:
            if 'metrics' in self.scheduler.step.__code__.co_varnames:
                self.scheduler.step(metrics=self.state.mean_loss['total'])
            else:
                self.scheduler.step()

    def INFERENCE(self, state):
        if self.model.recurrent_wrap:
            mix = self.state.data['input']
            self.state.output, self.state.hidden = self.model(self.state.data['input'])
        else:
            self.state.output = self.model(self.state.data['input'])
        n, c, *dims = self.state.output.shape

    def VALIDATION_START(self, state):
        self.state.phase = 'validation'
        self.model.eval()
        self.clear_state()

    def VALIDATION_END(self, state):
        self.clear_state()

    def CHECKPOINT(self, state):
        torch.save(self.model.state_dict(), state.logdir / f'checkpoint_model_{state.iteration}.pth')
        torch.save(self.optimizer.state_dict(), state.logdir / f'checkpoint_optim_{state.iteration}.state')

    def EPOCH_END(self, state):
        self.state.epoch += 1

    def EXECUTION_END(self, state):
        pass

    def run(self, *args, **kwargs):
        self._fire('START')
        while self.state.epoch < self.cfg.engine.num_epochs or self.state.terminate:  # EPOCH
            self._fire('EPOCH_START', f'Epoch #{self.state.epoch}: ')
            self._fire('TRAINING_START', f'Epoch #{self.state.epoch}: ')
            for dataloader_name, dataloader in self.dataloaders['train'].items():

                self.state.sub_phase = dataloader_name
                self.state.diversity = 0
                self.state.diversity_data = None
                self.state.diversity_out = None

                super_batch = 1
                if hasattr(dataloader, 'super_batch'):
                    super_batch = dataloader.super_batch

                self._fire('ITERATION_START', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}: ')
                for data in dataloader:    # ITERATION

                    self.state.data = data
                    self._fire('BATCH_LOAD')
                    self._fire('INFERENCE')
                    self._fire('LOSS_UPDATE', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}, batch #{self.state.loss_step+1}: ')
                    self._fire('BACKWARD')

                    diversity = np.sum(self.state.data['label'] == 2)
                    if diversity > self.state.diversity or self.state.diversity_data is None:
                        self.state.diversity = diversity
                        self.state.diversity_data = self.state.data
                        self.state.diversity_out = self.state.output

                    if self.state.loss_step >= super_batch:
                        self._fire('OPTIMIZATION')

                    if self.cfg.engine.debug_mode:
                        self.log('debug_mode', 'WARNING', 'Debug mode is enabled, abort after first training iteration.')
                        break

                if self.state.loss_step > 0:
                    self._fire('OPTIMIZATION')
            self._fire('TRAINING_END')

            self._fire('VALIDATION_START', f'Epoch #{self.state.epoch}, iter #{self.state.iteration-1}: ')
            for dataloader_name, dataloader in self.dataloaders['validation'].items():

                self.state.sub_phase = dataloader_name
                self.state.diversity = 0
                self.state.diversity_data = None
                self.state.diversity_out = None

                for data in dataloader:    # ITERATION
                    self.state.data = data
                    self._fire('BATCH_LOAD')
                    self._fire('INFERENCE')
                    self._fire('LOSS_UPDATE')

                        #~ diversity = np.sum(self.state.data['label'] == 2)
                        #~ if diversity > self.state.diversity or self.state.diversity_data is None:
                            #~ self.state.diversity = diversity
                            #~ self.state.diversity_data = self.state.data
                            #~ self.state.diversity_out = self.state.output
                    if self.cfg.engine.debug_mode:
                        self.log('debug_mode', 'WARNING', 'Debug modeis enabled, abort after first training iteration.')
                        break

            self._fire('VALIDATION_END', f'Epoch #{self.state.epoch}, iter #{self.state.iteration}: ')
            self._fire('EPOCH_END', f'Epoch #{self.state.epoch}: ')

            if self.cfg.engine.debug_mode:
                self.log('debug_mode', 'WARNING', 'Debug modeis enabled, abort after first epoch.')
                break
        self._fire('EXECUTION_END')
