import random
import random
import numpy
import numpy as np


def number2layernum(num):
    layer_map = {
        0 : 6,
        1 : 8,
        2 : 10,
        3 : 12
    }

    return layer_map.get(num, None)

def layernum2number(layernum):
    layer_map2 = {
        6 : 0,
        8 : 1,
        10 : 2,
        12 : 3
    }

    return layer_map2.get(layernum, None)

def number2headnum(num):

    head_map = {
        0 : 4,
        1 : 8,
        2 : 12
    }

    return head_map.get(num%3, None)

def number2ffnsize(num):

    ffn_map = {
        0 : 512,
        1 : 768,
        2 : 1024,
        3 : 3072
    }

    return ffn_map.get(num//3, None)

def layer_arc2number(head_num, ffn_size):

    head_map2 = {
        4 : 0,
        8 : 1,
        12 : 2
    }

    ffn_map2 = {
        512 : 0,
        768 : 1,
        1024 : 2,
        3072 : 3
    }

    a = head_map2.get(head_num, None)
    b = ffn_map2.get(ffn_size, None)
    
    return b*3 + a

def one_hot_to_index(one_hot_matrix):
    return np.array([np.where(r == 1)[0][0] for r in one_hot_matrix])

def index_to_one_hot(index_vector, C):
    return np.eye(C)[index_vector.reshape(-1)]

def encode2config(encode):
    '''
    input
    : one_hot [d,cmax]
    output
    : arc_config dict
    '''
    arc_index = one_hot_to_index(encode)
    arc_config = {
        'common': {}
    }
    arc_config['common']["bert_hidden_size"] = 768
    arc_config['common']["bert_layer_num"] = number2layernum(arc_index[0])
    layer_num = arc_config['common']["bert_layer_num"]
    for i in range(layer_num):
        index_str = 'layer'+str(i+1)
        arc_config[index_str] = {}
        arc_config[index_str]['bert_intermediate_size'] = number2ffnsize(arc_index[i+1])
        arc_config[index_str]['bert_head_num'] = number2headnum(arc_index[i+1])
    
    return arc_config

def config2encode(config):
    '''
    input
    : arc_config dict
    output
    : one_hot np.array [d,cmax]
    '''
    layer_map2 = {
        6 : 0,
        8 : 1,
        10 : 2,
        12 : 3
    }
    arc_index = []
    layer_num = config['common']["bert_layer_num"]
    arc_index.append(layer_map2.get(layer_num, None))
    for i in range(layer_num):
        index_str = 'layer'+str(i+1)
        head_num = config[index_str]['bert_head_num']
        ffn_size = config[index_str]['bert_intermediate_size']
        arc_index.append(layer_arc2number(head_num, ffn_size))
    for j in range(layer_num, 12):
        index_str = 'layer'+str(j+1)
        arc_index.append(random.choice(list(range(0,12))))
    
    arc_one_hot = index_to_one_hot(np.array(arc_index), 12)

    return arc_one_hot

def get_subbert_search_space():

    config = {
            "bert_layer_num_choice": [6,8,10,12],

            "bert_hidden_size_choice" : [768],

            "bert_intermediate_choice": [512,768,1024,3072],

            "bert_head_choice": [4,8,12]
    }

    return config


def sample_subbert_config(reset_rand_seed = False):
    if reset_rand_seed:
        random.seed(0)

    search_space = get_subbert_search_space()

    config = {
        'common': {}
    }

    config['common']["bert_layer_num"] = random.choice(search_space["bert_layer_num_choice"])
    config['common']["bert_hidden_size"] = random.choice(search_space["bert_hidden_size_choice"])

    for i in range(config['common']["bert_layer_num"]):
        index_str = 'layer'+str(i+1)
        config[index_str] = {}
        config[index_str]['bert_intermediate_size'] = random.choice(search_space["bert_intermediate_choice"])
        config[index_str]['bert_head_num'] = random.choice(search_space["bert_head_choice"])
    
    return config

def get_default_config(config):

    sample_config = {
        'common': {}
    }

    sample_config['common']["bert_hidden_size"] = config.hidden_size
    sample_config['common']['bert_layer_num'] = config.num_hidden_layers

    bert_intermediate_size = config.intermediate_size
    bert_head_num = config.num_attention_heads

    for i in range(12):
        index_str = 'layer'+str(i+1)
        sample_config[index_str] = {}
        sample_config[index_str]['bert_intermediate_size'] = bert_intermediate_size
        sample_config[index_str]['bert_head_num'] = bert_head_num

    return sample_config
    

def get_represent_config(k):
  
    default_config ={
        'common': {'bert_hidden_size': 768, 'bert_layer_num': 12}
    }
    for i in range(default_config['common']["bert_layer_num"]):
        index_str = 'layer'+str(i+1)
        default_config[index_str] = {}
        default_config[index_str]['bert_intermediate_size'] = 3072
        default_config[index_str]['bert_head_num'] = 12

    mid_config ={
        'common': {'bert_hidden_size': 768, 'bert_layer_num': 6}
    }
    
    for i in range(mid_config['common']["bert_layer_num"]):
        index_str = 'layer'+str(i+1)
        mid_config[index_str] = {}
        mid_config[index_str]['bert_intermediate_size'] = 3072
        mid_config[index_str]['bert_head_num'] = 12
    if k == 0:
        return default_config
    else: 
        return mid_config

def get_eval_config(train_type, train_kind, phase):

    eval_config = {}

    search_space = get_subbert_search_space()
    
    if train_type == 'head':
        if train_kind == 'all':
            search_space["bert_layer_num_choice"] = [12]
            search_space["bert_intermediate_choice"] = [3072]
        else:
            if phase == 0:
                search_space["bert_layer_num_choice"] = [12]
                search_space["bert_intermediate_choice"] = [3072]
                search_space["bert_head_choice"] = [8,12]
            else:
                search_space["bert_layer_num_choice"] = [12]
                search_space["bert_intermediate_choice"] = [3072]
                search_space["bert_head_choice"] = [4,8,12]
    elif train_type == 'ffn':
        if train_kind == 'ps':
            if phase == 0:
                search_space["bert_layer_num_choice"] = [12]
                search_space["bert_intermediate_choice"] = [1024,3072]
            elif phase == 1:
                search_space["bert_layer_num_choice"] = [12]
                search_space["bert_intermediate_choice"] = [768,1024,3072]
            else:
                search_space["bert_layer_num_choice"] = [12]
                search_space["bert_intermediate_choice"] = [512,768,1024,3072]
        else:
            search_space["bert_layer_num_choice"] = [12]
    else:
        if train_kind == 'ps':
            if phase == 0:
                search_space["bert_layer_num_choice"] = [10,12]
            elif phase == 1:
                search_space["bert_layer_num_choice"] = [8,10,12]
            else:
                search_space["bert_layer_num_choice"] = [6,8,10,12]
        else:
            pass

    for ra in range(10):
        config = {'common': {}}
        config['common']["bert_layer_num"] = random.choice(search_space["bert_layer_num_choice"])
        config['common']["bert_hidden_size"] = 768
        for i in range(config['common']["bert_layer_num"]):
            index_str = 'layer'+str(i+1)
            config[index_str] = {}
            config[index_str]['bert_intermediate_size'] = random.choice(search_space["bert_intermediate_choice"])
            config[index_str]['bert_head_num'] = random.choice(search_space["bert_head_choice"])

        eval_config['random'+str(ra)] = config 

    

    return eval_config

def make_biggest_dim(search_space , train_type):
    if train_type == 'head':
        search_space["bert_head_choice"] = [12]
    elif train_type == 'ffn':
        search_space["bert_intermediate_choice"] = [3072]    
    else:
        search_space["bert_layer_num_choice"] = [12]
    return search_space
        
def sample_train_config(train_type, train_kind, phase, train_seq, distribution_optimizer, prob, reset_rand_seed = False):
    
    print("prob",prob)
    sample_config = []

    search_space = get_subbert_search_space()
    
    index = train_seq.index(train_type)
    
    for i in range(index+1,len(train_seq)):
        t_type = train_seq[i]
        search_space = make_biggest_dim(search_space , t_type)
    
    if train_type == 'head':
        if train_kind != 'all':
            if phase == 0:
                search_space["bert_head_choice"] = [8,12]
            else:
                search_space["bert_head_choice"] = [4,8,12]
        else:
            search_space["bert_head_choice"] = [4,8,12]
    elif train_type == 'ffn':
        if train_kind == 'ps':
            if phase == 0:
                search_space["bert_intermediate_choice"] = [1024,3072]
            elif phase == 1:
                search_space["bert_intermediate_choice"] = [768,1024,3072]
            else:
                search_space["bert_intermediate_choice"] = [512,768,1024,3072]
        else:
            search_space["bert_intermediate_choice"] = [512,768,1024,3072]
    else:
        if train_kind == 'ps':
            if phase == 0:
                search_space["bert_layer_num_choice"] = [10,12]
            elif phase == 1:
                search_space["bert_layer_num_choice"] = [8,10,12]
            else:
                search_space["bert_layer_num_choice"] = [6,8,10,12]
        else:
            search_space["bert_layer_num_choice"] = [6,8,10,12]

    if random.random() < prob:
        sample_one_hot = distribution_optimizer.sampling_subspace(search_space)
        sample_subbert = encode2config(sample_one_hot)
        sample_config.append(sample_subbert)
    else:
        config = {'common': {}}
        config['common']["bert_layer_num"] = random.choice(search_space["bert_layer_num_choice"])
        config['common']["bert_hidden_size"] = 768
        for i in range(config['common']["bert_layer_num"]):
            index_str = 'layer'+str(i+1)
            config[index_str] = {}
            config[index_str]['bert_intermediate_size'] = random.choice(search_space["bert_intermediate_choice"])
            config[index_str]['bert_head_num'] = random.choice(search_space["bert_head_choice"])
        sample_config.append(config)
    return sample_config

def sample_train_config_exploit(train_type, train_kind, phase, train_seq, distribution_optimizer, prob, reset_rand_seed = False):
    prob = 1
    
    sample_config = []

    search_space = get_subbert_search_space()
    
    index = train_seq.index(train_type)
    
    for i in range(index+1,len(train_seq)):
        t_type = train_seq[i]
        search_space = make_biggest_dim(search_space , t_type)
    
    if train_type == 'head':
        if train_kind != 'all':
            if phase == 0:
                search_space["bert_head_choice"] = [8,12]
            else:
                search_space["bert_head_choice"] = [4,8,12]
        else:
            search_space["bert_head_choice"] = [4,8,12]
    elif train_type == 'ffn':
        if train_kind == 'ps':
            if phase == 0:
                search_space["bert_intermediate_choice"] = [1024,3072]
            elif phase == 1:
                search_space["bert_intermediate_choice"] = [768,1024,3072]
            else:
                search_space["bert_intermediate_choice"] = [512,768,1024,3072]
        else:
            search_space["bert_intermediate_choice"] = [512,768,1024,3072]
    else:
        if train_kind == 'ps':
            if phase == 0:
                search_space["bert_layer_num_choice"] = [10,12]
            elif phase == 1:
                search_space["bert_layer_num_choice"] = [8,10,12]
            else:
                search_space["bert_layer_num_choice"] = [6,8,10,12]
        else:
            search_space["bert_layer_num_choice"] = [6,8,10,12]

    if random.random() < prob:
        sample_one_hot = distribution_optimizer.sampling_subspace(search_space)
        sample_subbert = encode2config(sample_one_hot)
        sample_config.append(sample_subbert)
    else:
        config = {'common': {}}
        config['common']["bert_layer_num"] = random.choice(search_space["bert_layer_num_choice"])
        config['common']["bert_hidden_size"] = 768
        for i in range(config['common']["bert_layer_num"]):
            index_str = 'layer'+str(i+1)
            config[index_str] = {}
            config[index_str]['bert_intermediate_size'] = random.choice(search_space["bert_intermediate_choice"])
            config[index_str]['bert_head_num'] = random.choice(search_space["bert_head_choice"])
        sample_config.append(config)
    return sample_config

def sample_train_config_explore(train_type, train_kind, phase, train_seq, distribution_optimizer, prob, reset_rand_seed = False):
    prob = 0
    
    sample_config = []

    search_space = get_subbert_search_space()
    
    index = train_seq.index(train_type)
    
    for i in range(index+1,len(train_seq)):
        t_type = train_seq[i]
        search_space = make_biggest_dim(search_space , t_type)
    
    if train_type == 'head':
        if train_kind != 'all':
            if phase == 0:
                search_space["bert_head_choice"] = [8,12]
            else:
                search_space["bert_head_choice"] = [4,8,12]
        else:
            search_space["bert_head_choice"] = [4,8,12]
    elif train_type == 'ffn':
        if train_kind == 'ps':
            if phase == 0:
                search_space["bert_intermediate_choice"] = [1024,3072]
            elif phase == 1:
                search_space["bert_intermediate_choice"] = [768,1024,3072]
            else:
                search_space["bert_intermediate_choice"] = [512,768,1024,3072]
        else:
            search_space["bert_intermediate_choice"] = [512,768,1024,3072]
    else:
        if train_kind == 'ps':
            if phase == 0:
                search_space["bert_layer_num_choice"] = [10,12]
            elif phase == 1:
                search_space["bert_layer_num_choice"] = [8,10,12]
            else:
                search_space["bert_layer_num_choice"] = [6,8,10,12]
        else:
            search_space["bert_layer_num_choice"] = [6,8,10,12]

    if random.random() < prob:
        sample_one_hot = distribution_optimizer.sampling_subspace(search_space)
        sample_subbert = encode2config(sample_one_hot)
        sample_config.append(sample_subbert)
    else:
        config = {'common': {}}
        config['common']["bert_layer_num"] = random.choice(search_space["bert_layer_num_choice"])
        config['common']["bert_hidden_size"] = 768
        for i in range(config['common']["bert_layer_num"]):
            index_str = 'layer'+str(i+1)
            config[index_str] = {}
            config[index_str]['bert_intermediate_size'] = random.choice(search_space["bert_intermediate_choice"])
            config[index_str]['bert_head_num'] = random.choice(search_space["bert_head_choice"])
        sample_config.append(config)
    return sample_config

def sample_random_config():
    
    search_space = get_subbert_search_space()

    config = {
        'common': {}
    }

    config['common']["bert_layer_num"] = random.choice(search_space["bert_layer_num_choice"])
    config['common']["bert_hidden_size"] = random.choice(search_space["bert_hidden_size_choice"])

    for i in range(config['common']["bert_layer_num"]):
        index_str = 'layer'+str(i+1)
        config[index_str] = {}
        config[index_str]['bert_intermediate_size'] = random.choice(search_space["bert_intermediate_choice"])
        config[index_str]['bert_head_num'] = random.choice(search_space["bert_head_choice"])
        
    sample_config = []
    
    sample_config.append(config)
    return sample_config

def get_random_eval_config():
    
    search_space = get_subbert_search_space()
    eval_config = {}
    for ra in range(10):
        config = {'common': {}}
        config['common']["bert_layer_num"] = random.choice(search_space["bert_layer_num_choice"])
        config['common']["bert_hidden_size"] = 768
        for i in range(config['common']["bert_layer_num"]):
            index_str = 'layer'+str(i+1)
            config[index_str] = {}
            config[index_str]['bert_intermediate_size'] = random.choice(search_space["bert_intermediate_choice"])
            config[index_str]['bert_head_num'] = random.choice(search_space["bert_head_choice"])

        eval_config['random'+str(ra)] = config 
        
    return eval_config