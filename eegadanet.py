import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pickle



def get_norm(norm_type):
    def norm(c, norm_type):   
        if norm_type=='BatchNorm':
            return nn.BatchNorm1d(c)
        elif norm_type=='InstanceNorm':
            return nn.InstanceNorm1d(c)
        elif norm_type=='InstanceNormAffine':
            return nn.InstanceNorm1d(c, affine=True)
        else:
            return nn.Identity()
    return partial(norm, norm_type=norm_type)


def get_act(act_type):
    if act_type=='gelu':
        return nn.GELU()
    elif act_type=='silu':
        return nn.SiLU()
    elif act_type=='relu':
        return nn.ReLU()
    elif act_type=='lrelu':
        return nn.LeakyReLU(0.1)
    else:
        raise Exception



class FourierEmb(nn.Module):
    """
    3D extension of the periodic fourier embeddings described in 
    the paper "Decoding speech perception from non-invasive brain recordings"
    (github.com/facebookresearch/brainmagick/blob/main/bm/models/common.py#L239)
    """
    def __init__(self, n_freqs, margin=0.2):
        super().__init__()
        self.n_freqs = n_freqs
        self.dimension = n_freqs**3 * 2
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        freqs_z = torch.arange(self.n_freqs).to(positions)
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_z[:, None, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * np.pi * freqs_x / width
        p_y = 2 * np.pi * freqs_y / width
        p_z = 2 * np.pi * freqs_z / width
        positions = positions[..., None, None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y + positions[..., 2] * p_z).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb


class SpatialAttention(nn.Module):
    def __init__(self, dim, att_dim, n_head, att_drop):
        super().__init__()
 
        self.Q = nn.Linear(dim, att_dim, bias=False)
        self.K = nn.Linear(dim, att_dim, bias=False)

        self.C = nn.Conv1d(dim*2, dim, 1, 1, 0, bias=False)

        self.drop = nn.Dropout(att_drop)
        
        self.num_heads = n_head

        # store attention matrices
        self.matrix = None

    def to_multihead(self, x):
        x = x.reshape(*x.shape[:-1], self.num_heads, -1)
        return x.transpose(1, 2)

    def gather_heads(self, x):
        x = x.transpose(1, 2)
        return x.reshape(*x.shape[:-2], -1)

    # qk.shape = [batch, sequence, features]
    # x.shape = [batch, sequence, features, ...]
    def attention(self, q, k, x, mask, save_matrix=False):

        input_shape = x.shape

        q = self.to_multihead(q)
        k = self.to_multihead(k)
        x = self.to_multihead(x.reshape(*x.shape[:2],-1))

        a = torch.matmul(q, k.transpose(-1, -2)) / k.shape[-1]**0.5
        
        if mask is not None:
            a = a + mask[:,None,None]
            
        a = self.drop(torch.softmax(a, dim=-1))

        if save_matrix:
            self.matrix = a
        
        x = torch.matmul(a, x)

        x = self.gather_heads(x).reshape(input_shape)

        return x

    # x.shape = [batch*channels, features, time]
    def forward(self, x, emb, att_mask, save_matrix=False):   

        residual = x

        x = x.reshape(-1,len(emb),*x.shape[-2:])

        q, k = self.Q(x.mean(-1)), self.K(x.mean(-1))

        x = self.attention(q, k, x, att_mask, save_matrix)

        x = x.reshape(-1,*x.shape[-2:])
        
        x = self.C(torch.cat([x, residual],1))

        return x


# input shape = [batch, features, time]
class FiLM(nn.Module):
    def __init__(self, num_channels, condition_dim):        
        super().__init__()

        self.cond_proj = nn.Linear(condition_dim, 2*num_channels)
    
    def forward(self, x, c):
        c = c.repeat(len(x)//len(c),1)
        weight, bias = torch.chunk(self.cond_proj(c), 2, dim=-1)
        x = x * weight[...,None] + bias[...,None]
        return x


class TCB(nn.Module):
    def __init__(self, c, l, k, norm, activation, cond_dim):        
        super().__init__()

        self.film = FiLM(c, cond_dim)

        self.layers = nn.ModuleList()
        for i in range(l): 
            conv = nn.Sequential(
                norm(c),
                activation,
                nn.Conv1d(c, c, k, 1, k//2, bias=False)
            )
            self.layers.append(conv)
    
    def forward(self, x, c):
        s = x
        for layer in self.layers:
            x = layer(x)
        x = self.film(x, c)
        x = x + s
        return x

            
    
class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

        k = cfg.kernel_size

        # model depth
        N = cfg.num_blocks
        self.n = len(N) - 1
        
        # num channels for each scale
        scale = cfg.scale
        C = cfg.num_channels
        
        norm = get_norm(norm_type=cfg.norm)
        activation = get_act(act_type=cfg.activation)

        if cfg.emb_freqs > 0:
            self.electrode_embedding = FourierEmb(cfg.emb_freqs)
        else:
            self.electrode_embedding = None
        F = self.electrode_embedding.dimension

        self.first_conv = nn.Conv1d(1, C, 3, 1, 1, bias=False)

        self.down_blocks = nn.ModuleList()
        for i in range(self.n):
            block = nn.Module()
            block.tcb = TCB(C, N[i], k, norm, activation, F)
            block.att = SpatialAttention(C, cfg.att_dim, cfg.att_heads, cfg.att_drop)
            block.downsample = nn.Conv1d(C, C, scale, scale, bias=False)   
            self.down_blocks.append(block)    

        self.bottleneck_block = nn.Module()
        self.bottleneck_block.tcb = TCB(C, N[-1], k, norm, activation, F)
        self.bottleneck_block.att = SpatialAttention(C, cfg.att_dim, cfg.att_heads, cfg.att_drop)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.n)):    
            block = nn.Module()
            block.upsample = nn.ConvTranspose1d(C, C, scale, scale, bias=False) 
            block.tcb = TCB(C, N[i], k, norm, activation, F)
            block.att = SpatialAttention(C, cfg.att_dim, cfg.att_heads, cfg.att_drop)
            self.up_blocks.append(block)     
       
        self.final_conv = nn.Conv1d(C, 1, 1, 1, 0, bias=False)

    # x.shape = [batch, channels, time]
    # coords.shape = [channels, 3]
    # channel_mask.shape = [batch, channels]
    def forward(self, x, coords, channel_mask=None, save_matrix=False):

        emb = self.electrode_embedding(coords).to(x.device)
        
        if channel_mask is None:
            att_mask = None
        else:
            att_mask = torch.zeros_like(channel_mask.float()) \
                            .masked_fill_(~channel_mask, float('-inf'))
        
        B,C,T = x.shape
        
        sigma = x.std(-1, keepdim=True)
        x = x / sigma
        x = x.reshape(-1,T)[:,None]

        x = self.first_conv(x)
        
        D = []
        for i, block in enumerate(self.down_blocks): 
            x = block.tcb(x, emb) 
            x = block.att(x, emb, att_mask, save_matrix)
            D.append(x)
            x = block.downsample(x)

        x = self.bottleneck_block.tcb(x, emb)
        x = self.bottleneck_block.att(x, emb, att_mask, save_matrix)

        for i, block in enumerate(self.up_blocks):    
            x = block.upsample(x)
            x = x + D.pop()
            x = block.tcb(x, emb) 
            x = block.att(x, emb, att_mask, save_matrix)
        
        x = self.final_conv(x)

        x = x[:,0].reshape(B,C,T)
        
        return x * sigma
        