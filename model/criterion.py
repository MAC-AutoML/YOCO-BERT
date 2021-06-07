import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter


class criterion_compute(nn.Module):
    def __init__(self, t_config, s_config):
        super(criterion_compute, self).__init__()
        self.t_config = t_config
        self.s_config = s_config

    def forward(self, t_model, s_model, input_ids, token_type_ids, attention_mask, labels, args):
        with torch.no_grad():
            t_outputs = t_model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)

        s_outputs = s_model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels)

        t_logits, t_features = t_outputs[0], t_outputs[-1]
        train_loss, s_logits, s_features = s_outputs[0], s_outputs[1], s_outputs[-1]
        T = args.temperature
        soft_targets = F.softmax(t_logits / T, dim=-1)
        log_probs = F.log_softmax(s_logits / T, dim=-1)
        soft_loss = F.kl_div(log_probs, soft_targets.detach(), reduction='batchmean') * T * T

        return train_loss, soft_loss, 0
