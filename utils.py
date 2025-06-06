import os
from os.path import join
import random
import numpy as np
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from tqdm.notebook import tqdm
from glob import glob
from functools import partial
import mne
import cv2
import matplotlib.pyplot as plt


class EarlyStop():
    def __init__(self, config):
        self.config = config
        self.loss_per_epoch = []
        self.best_epoch = 0
        self.best_ckpt = None
        self.stop = False

    def update(self, loss):
        self.loss_per_epoch.append(loss)
        if self.best_epoch==0:
            self.best_epoch = 1
        else:
            interval = self.loss_per_epoch[self.best_epoch-1:]
            if loss + self.config.tolerance <= interval[0]:
                self.best_epoch = len(self.loss_per_epoch)
            elif len(interval)==self.config.patience:
                self.stop = True
        

def flatten(xss):
    return [x for xs in xss for x in xs]
    

def mkdir(p, is_file=False):
    if is_file:
        p = os.path.dirname(p)
    if not os.path.isdir(p):
        os.makedirs(p)
    

def manual_seed(seed, dil=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = not dil
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    
def num_params(model, include_frozen=False):
    n = 0
    for param in model.parameters():
        if (not include_frozen and param.requires_grad) or include_frozen:
            n += param.numel()
    return round(n/1e6, 2)


def no_grad(model):
    for p in model.parameters():
        p.requires_grad = False


@torch.no_grad()
def get_flops(model, x):
    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        model(x)
    return flop_counter.get_total_flops()


def power_ratio(a, b):
    return (a**2).sum(-1)/((b**2).sum(-1)+1e-8)

def SNR(signal, noise):
    return 10*np.log10(power_ratio(signal, noise))

def RRMSE(x_, x):
    return power_ratio(x_-x, x)**0.5


def PSD(raw, selected_channel, fmin=0, fmax=np.inf, n_fft=2048, interval=[None,None]):
    tmin, tmax = interval
    psd = raw.compute_psd(
        picks=[selected_channel], 
        tmin=tmin, tmax=tmax,
        fmin=fmin, fmax=fmax, n_fft=n_fft,
        verbose='error').get_data()[0]
    psd *= 1e6**2
    psd = 10*np.log10(np.maximum(psd, np.finfo(float).tiny), out=psd)
    return psd


def tensor2raw(x, sr, electrodes, scale=1, ch_types=None):
    if ch_types is None:
        ch_types=['eeg']*len(electrodes)
    return mne.io.RawArray(
        x.numpy()/1e6 * scale,
        mne.create_info(
            electrodes, 
            sr, 
            ch_types=ch_types
        ),
        verbose='error'
    )


def plot_chart(raw, color, size, save_path):
    fig = raw.plot(
        duration=raw.n_times//raw.info['sfreq'],
        show_scrollbars=False,
        show_scalebars=False,
        show=False,
        color=dict(eeg=color, eog='orange')
    )
    fig.set_size_inches(*size)
    fig.savefig(save_path)
    return cv2.imread(save_path)


def plot_overlayed_chart(a, b, save_path, size=[10,5], color='orange'):

    a = plot_chart(a, color, size, os.path.dirname(save_path)+'/a.png')
    b = plot_chart(b, 'black', size, os.path.dirname(save_path)+'/b.png')
    plt.show()

    pixel_mean = b.mean(-1,keepdims=True) / 255
    ab = (pixel_mean>0.99) * a + b
    cv2.imwrite(save_path, ab)



@torch.no_grad()  
def chunk_and_apply(f, x, batch_size, chunk_size, overlap_factor=None, single_channel=False):
    
    C, T = x.shape

    if overlap_factor is not None:
        assert chunk_size%overlap_factor==0
        overlap = chunk_size//overlap_factor
        x = torch.cat([torch.zeros(*batch_dims, overlap).to(x.device), x], -1)
    else:
        overlap = 0
        
    gen_size = chunk_size - 2*overlap
    pad_size = gen_size - T%gen_size
    
    x = torch.cat([x, torch.zeros(C, pad_size+overlap).to(x.device)], -1)
    
    chunks = []
    i = 0
    while i < T + pad_size:
        chunk = x[..., i:i+chunk_size]
        chunks.append(chunk)
        i += gen_size
    chunks = torch.stack(chunks)
    
    batches = []
    i = 0
    while i < len(chunks):
        batches.append(chunks[i:i+batch_size])
        i = i + batch_size
        
    output_chunks = []   
    for batch in batches:
        
        if single_channel:
            batch = batch.reshape(-1,chunk_size)
            
        x = f(x=batch) 
        
        if single_channel:
            x = x.reshape(-1,C,chunk_size)
            
        if overlap > 0:  
            x = x[...,overlap:-overlap]  
            
        output_chunks += torch.unbind(x.cpu(), 0)  

    return torch.cat(output_chunks, -1)[...,:-pad_size]




        
        

   
    

