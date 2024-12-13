import os
from os.path import join
import random
import numpy as np
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from glob import glob
from functools import partial


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


def power_ratio(a, b):
    return (a**2).sum(-1)/((b**2).sum(-1)+1e-8)


def SNR(signal, noise):
    return 10*np.log10(power_ratio(signal, noise))


@torch.no_grad()  
def chunk_and_apply(f, x, batch_size, chunk_size, overlap_factor=None):
    
    batch_dims, n_samples = x.shape[:-1], x.shape[-1]

    if overlap_factor is not None:
        assert chunk_size%overlap_factor==0
        overlap = chunk_size//overlap_factor
        x = torch.cat([torch.zeros(*batch_dims, overlap).to(x.device), x], -1)
    else:
        overlap = 0
        
    gen_size = chunk_size - 2*overlap
    pad_size = gen_size - n_samples%gen_size
    
    x = torch.cat([x, torch.zeros(*batch_dims, pad_size+overlap).to(x.device)], -1)
    
    chunks = []
    i = 0
    while i < n_samples + pad_size:
        chunk = x[:, i:i+chunk_size]
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
        x = f(x=batch) 
        if overlap > 0:  
            x = x[...,overlap:-overlap]        
        output_chunks += torch.unbind(x.cpu(), 0)  

    return torch.cat(output_chunks, -1)[...,:-pad_size]




        
        

   
    

