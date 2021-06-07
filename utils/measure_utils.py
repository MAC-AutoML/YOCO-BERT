import torchprofile
import torch
import random
import numpy as np

def measure_flops(model, config, dummy_input):

    model.set_sample_config(config)
    model.profile(mode=True)
    macs = torchprofile.profile_macs(model, dummy_input)
    model.profile(mode=False)
    return macs*2

def get_dummy_input(args, device_using):

    dummy_input = (torch.tensor([[5000]*64+[0]*64]).to(device_using), # input_id
                   torch.tensor([[1]*64+[0]*64]).to(device_using), # attention_mask
                   torch.tensor([[0]*50+[1]*28+[0]*50]).to(device_using), # token_type_ids
                   torch.tensor([[random.choice([0,1])]]).to(device_using)) # labels

    return dummy_input

def get_feature_info():
    
    return ['bert_layer_num','bert_hidden_size','bert_intermediate_avg','bert_head_avg']

def get_flops_feature(config):
    
    feature = []

    feature.append(config['common']['bert_layer_num'])
    feature.append(config['common']['bert_hidden_size'])
    

    ffn_sum = []
    for i in range(config['common']["bert_layer_num"]):
        index_str = 'layer'+str(i+1)
        ffn_sum.append(config[index_str]['bert_intermediate_size'])
    feature.append(np.mean(ffn_sum))


    head_sum = []
    for i in range(config['common']["bert_layer_num"]):
        index_str = 'layer'+str(i+1)
        head_sum.append(config[index_str]['bert_head_num'])
    feature.append(np.mean(head_sum))
    
    return feature