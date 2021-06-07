import torch
import torch.nn.functional as F

class SuperLayerNorm(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim, eps):
        super().__init__(super_embed_dim, eps)

        self.super_embed_dim = super_embed_dim
  
        self.sample_embed_dim = None
        self.samples = {}

        self.profiling = False

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(x, (self.sample_embed_dim,), weight = self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def profile(self, mode=True):
        self.profiling = mode