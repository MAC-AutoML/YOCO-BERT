import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SuperEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, super_embed_dim, padding_idx = None, *args, **kwargs):
        super().__init__(num_embeddings, super_embed_dim, padding_idx, *args, **kwargs)


        self.super_embed_dim = {'encoder': super_embed_dim}

   
        self.sample_embed_dim = {'encoder': None}

        self.samples = {'encoder': {}}
        self.reset_parameters()


        self.profiling = False

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.weight[self.padding_idx], 0)

    def set_sample_config(self, sample_embed_dim, part = 'encoder'):
        self.sample_embed_dim[part] = sample_embed_dim
        self._sample_parameters(part)

    def _sample_parameters(self, part):
        weight = self.weight[..., :self.sample_embed_dim[part]]
        self.samples[part]['weight'] = weight

        return self.samples

    def sample_parameters(self, part, resample=False):
        return self._sample_parameters(part) if self.profiling or resample else self.samples

    def sampled_weight(self, part):
        return self.sample_parameters(part)[part]['weight']

    def forward(self, input, part='encoder'):
        return F.embedding(input, self.sampled_weight(part), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def profile(self, mode=True):
        self.profiling = mode
        
    def calc_sampled_param_num(self):
        return self.samples['encoder']['weight'].numel()  