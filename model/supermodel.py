# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

import transformers
from transformers.activations import ACT2FN
from transformers.configuration_bert import BertConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_bert import load_tf_weights_in_bert  
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

import transformers
from transformers.activations import ACT2FN
from transformers.configuration_bert import BertConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_bert import load_tf_weights_in_bert  
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from modules.embeddingsuper import SuperEmbedding
from modules.layernormsuper import SuperLayerNorm
from modules.linearsuper import SuperLinear

class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = SuperEmbedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = SuperEmbedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = SuperEmbedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = SuperLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.sample_hidden_size = None

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None): 
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def set_sample_config(self, bert_hidden_size):

        self.sample_hidden_size = bert_hidden_size

        self.word_embeddings.set_sample_config(bert_hidden_size)
        self.position_embeddings.set_sample_config(bert_hidden_size)
        self.token_type_embeddings.set_sample_config(bert_hidden_size)
        self.LayerNorm.set_sample_config(bert_hidden_size)

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = SuperLinear(config.hidden_size, self.all_head_size)
        self.key = SuperLinear(config.hidden_size, self.all_head_size)
        self.value = SuperLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sample_hidden_size = None
        self.sample_num_attention_heads = None 
        self.sample_attention_head_size = None 
        self.sample_all_head_size = None 

        self.super_hidden_size = config.hidden_size

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.sample_num_attention_heads, self.sample_attention_head_size) 
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) 

    def forward(
        self,
        hidden_states,
        attention_mask=None, 
        head_mask=None, 
        encoder_hidden_states=None, 
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

    
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) 
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
      
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores) 


        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) 

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
        new_context_layer_shape = context_layer.size()[:-2] + (self.sample_all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs 


    def set_sample_config(self, bert_hidden_size, bert_head_num):

        self.sample_hidden_size = bert_hidden_size
        self.sample_num_attention_heads = bert_head_num

        self.sample_attention_head_size = self.attention_head_size
        self.sample_all_head_size = self.sample_num_attention_heads * self.sample_attention_head_size

        self.query.set_sample_config(sample_in_dim = self.sample_hidden_size, sample_out_dim = self.sample_all_head_size)
        self.key.set_sample_config(sample_in_dim = self.sample_hidden_size, sample_out_dim = self.sample_all_head_size)
        self.value.set_sample_config(sample_in_dim = self.sample_hidden_size, sample_out_dim = self.sample_all_head_size)

class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = SuperLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = SuperLayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        self.sample_hidden_size = None
        self.sample_head_num = None

        self.origin_num_attention_heads = config.num_attention_heads
        self.origin_attention_head_size = int(config.hidden_size / config.num_attention_heads)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def set_sample_config(self, bert_hidden_size, bert_head_num):

        self.sample_hidden_size = bert_hidden_size
        self.sample_head_num = bert_head_num
        self.sample_all_head_size = self.origin_attention_head_size * self.sample_head_num

        self.dense.set_sample_config(self.sample_all_head_size, self.sample_hidden_size)
        self.LayerNorm.set_sample_config(self.sample_hidden_size)


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()


        self.sample_hidden_size = None
        self.sample_num_attention_heads = None 

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )


        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)


        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

       

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states) 
        outputs = (attention_output,) + self_outputs[1:] 
        return outputs

    def set_sample_config(self, bert_hidden_size, bert_head_num):

        self.sample_hidden_size = bert_hidden_size
        self.sample_num_attention_heads= bert_head_num

        self.self.set_sample_config(self.sample_hidden_size, self.sample_num_attention_heads)
        self.output.set_sample_config(self.sample_hidden_size, self.sample_num_attention_heads)

class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = SuperLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        

        self.sample_hidden_size = None 
        self.sample_intermediate_size = None 

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def set_sample_config(self, bert_hidden_size, bert_intermediate_size):

        self.sample_hidden_size = bert_hidden_size
        self.sample_intermediate_size = bert_intermediate_size

        self.dense.set_sample_config(self.sample_hidden_size, self.sample_intermediate_size)




class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = SuperLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = SuperLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.sample_hidden_size = None 
        self.sample_intermediate_size = None 

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def set_sample_config(self, bert_hidden_size, bert_intermediate_size):

        self.sample_hidden_size = bert_hidden_size
        self.sample_intermediate_size = bert_intermediate_size

        self.dense.set_sample_config(self.sample_intermediate_size, self.sample_hidden_size)
        self.LayerNorm.set_sample_config(self.sample_hidden_size)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1 
        self.attention = BertAttention(config)
        
  
        self.is_decoder = config.is_decoder 
        self.add_cross_attention = config.add_cross_attention  
        if self.add_cross_attention: 
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


        self.sample_hidden_size = None 
        self.sample_intermediate_size = None 
        self.sample_num_attention_heads = None 

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0] 
        outputs = self_attention_outputs[1:]  

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs 

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


    def set_sample_config(self, bert_hidden_size, bert_intermediate_size, bert_head_num):

        self.sample_hidden_size = bert_hidden_size
        self.sample_intermediate_size = bert_intermediate_size
        self.sample_num_attention_heads = bert_head_num

        self.attention.set_sample_config(self.sample_hidden_size, self.sample_num_attention_heads)
        if self.add_cross_attention: 
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention.set_sample_config(self.sample_hidden_size, self.sample_num_attention_heads)
        self.intermediate.set_sample_config(self.sample_hidden_size, self.sample_intermediate_size)
        self.output.set_sample_config(self.sample_hidden_size, self.sample_intermediate_size)



class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])


        self.sample_config = None
        self.sample_num_layer = None 
        self.sample_hidden_size = None
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for i in range(self.sample_num_layer):
            layer_module = self.layer[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_self_attentions,)
        return outputs  

    def set_sample_config(self, sample_config):
        self.sample_config = sample_config
        self.sample_num_layer = sample_config['common']["bert_layer_num"]
        self.sample_hidden_size = sample_config['common']["bert_hidden_size"]

        for i in range(self.sample_num_layer):
            tmp_layer = self.layer[i]
            index_str = 'layer'+str(i+1)

            sample_intermediate_size = sample_config[index_str]['bert_intermediate_size'] 
            sample_num_attention_heads = sample_config[index_str]['bert_head_num'] 

            tmp_layer.set_sample_config(self.sample_hidden_size, sample_intermediate_size, sample_num_attention_heads)

        



class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = SuperLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


        self.sample_hidden_size = None
    def forward(self, hidden_states):

        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def set_sample_config(self, bert_hidden_size):

        self.sample_hidden_size = bert_hidden_size

        self.dense.set_sample_config(self.sample_hidden_size, self.sample_hidden_size)

        

class BertPreTrainedModel(PreTrainedModel):


    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (SuperEmbedding, SuperLinear)):
 
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, SuperLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, SuperLinear) and module.bias is not None:
            module.bias.data.zero_()




class BertModel(BertPreTrainedModel):


    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()


        self.sample_config = None
        self.sample_hidden_size = None


        self.add_pooling_layer = add_pooling_layer
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):

        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)


        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None


        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs
    def set_sample_config(self, sample_config):
        
        self.sample_config = sample_config
        self.sample_hidden_size = sample_config['common']["bert_hidden_size"]

        self.embeddings.set_sample_config(self.sample_hidden_size)
        self.encoder.set_sample_config(self.sample_config)
        if self.add_pooling_layer:
            self.pooler.set_sample_config(self.sample_hidden_size) 


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = SuperLinear(config.hidden_size, config.num_labels)
    
        self.init_weights()

        self.sample_config = None
        self.sample_hidden_size = None


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,

        output_attentions=None,
        output_hidden_states=None,
    ):


        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  

        loss = None
        if labels is not None:
            if self.num_labels == 1:
    
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs 
    def set_sample_config(self, sample_config):

        self.sample_config = sample_config
        self.sample_hidden_size = sample_config['common']["bert_hidden_size"]

        self.bert.set_sample_config(self.sample_config)
        self.classifier.set_sample_config(self.sample_hidden_size, self.num_labels)

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
 
                if name == 'classifier':
                    continue
   
                if name.split('.')[1] == 'encoder' and eval(name.split('.')[3]) + 1 > config['common']['bert_layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())
        return sum(numels)
        
    def profile(self, mode = True):
        for module in self.modules():
            if hasattr(module, 'profile') and self != module:
                module.profile(mode)

