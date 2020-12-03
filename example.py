#!/usr/bin/env python3

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
for dirname in os.environ.get('EXTRA_PYTHON_LIBS', '.').split(';'):
    sys.path.append(f'{Path(__file__).parent/dirname}/')
import flashlight
from flashlight.util import dcns
from loguru import logger
import skimage
from skimage.morphology import binary_dilation, disk

def contours(image):
    dilation = binary_dilation(image, disk(3))
    return dilation * ~image


engine = flashlight.auto_init()

@engine.on('AFTER_LOSS_UPDATE')
def log_print(state):
    logger.success(f'Epoch #{state.epoch} it. #{state.iteration}: {dcns(state.mean_loss)}')

@engine.on('BEFORE_OPTIMIZATION')
def log_to_prompt(state):

    prediction = np.argmax(state.output[0,...],axis=0)
    label = state.data['label'][0,...]

    label = one_hot(label.astype(np.uint8),3)
    prediction = one_hot(prediction.astype(np.uint8),3)

    a=terminal_art(np.rot90(prediction),'jet')
    b=terminal_art(np.rot90(label),'jet')

    result = []
    for x,y in zip(a,b):
        result.append(f'{x}   {y}')

    result = '\n'.join(result)
    logger.success(f'Epoch #{state.epoch} it. #{state.iteration}: current prediction vs label:\n' +result)


@engine.on('AFTER_LOSS_UPDATE')
def log_current(state):

    label_lesions = np.rot90(state.data['label'][0,...])==2
    predicted_lesions = np.rot90(np.argmax(state.output[0,...], axis=0)==2)

    plt.clf()
    K=state.data['input'].shape[1] //2 +1
    img = np.rot90(state.data['input'][0,K,...])
    plt.imshow(img,cmap='gray')
    plt.contour(label_lesions *1, levels=[.5,], colors='g', linestyles='-')
    plt.savefig(str(state.logdir /f'ground_truth_supervised_{state.iteration:04d}.png'))
    logger.success('Grount truth is logged.')
    plt.clf()

    plt.imshow(img,cmap='gray')
    plt.contour(predicted_lesions *1, levels=[.5,], colors='r', linestyles='-')
    plt.savefig(str(state.logdir /f'prediction_supervised_{state.iteration:04d}.png'))
    plt.clf()

@engine.on(('AFTER_OPTIMIZATION','BEFORE_VALIDATION_END'))
def log_plot(state):

    if state.diversity_data is not None:
        logger.info('image plot')
        fig = plt.figure(figsize=(8, 12))
        gs = gridspec.GridSpec(3, 2, hspace=0.15, wspace=0.15, left=0.03, right=0.97, bottom=0.03, top=0.98)

        ax00 = plt.subplot(gs[0, 0])
        ax01 = plt.subplot(gs[0, 1])
        ax10 = plt.subplot(gs[1, 0])
        ax11 = plt.subplot(gs[1, 1])
        ax20 = plt.subplot(gs[2, 0])
        ax21 = plt.subplot(gs[2, 1])

        x = state.diversity_data['label'].shape[1] // 2
        label = np.rot90(state.diversity_data['label'][0,x,...])
        pred = np.rot90(np.argmax(state.diversity_out[0,...], axis=0)[x,...])
        i = state.diversity_data['input'].shape[1] // 2
        data = np.rot90(state.diversity_data['input'][0,i, ...])

        ax00.imshow(data, cmap=plt.cm.gray)
        ax00.set_title('raw data')

        ax01.imshow(data, cmap=plt.cm.gray)
        ax01.set_title('raw data')

        ax10.imshow(label, cmap=plt.cm.gist_rainbow)
        ax10.set_title('ground truth')

        ax11.imshow(pred, cmap=plt.cm.gist_rainbow)
        ax11.set_title('prediction')

        ax20.imshow(data, cmap=plt.cm.gray)
        ax21.imshow(data, cmap=plt.cm.gray)
        ax20.imshow(label, alpha=0.5, cmap=plt.cm.gist_rainbow)
        ax21.imshow(pred, alpha=0.5, cmap=plt.cm.gist_rainbow)
        ax20.set_title('ground truth + overlay')
        ax21.set_title('prediction + overlay')
        plt.savefig(state.logdir /f'{state.phase}_{state.sub_phase}_{state.iteration:04d}.png')
        state.summary_writers[state.phase][state.sub_phase].add_figure(f'plot/prediction_label', fig, state.iteration)
    state.diversity_data = None

engine.run()
