

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
from yacs.config import CfgNode as CfgNode

# Global config object
_C = CfgNode()

args = _C


logger = logging.getLogger()
logger.setLevel(logging.DEBUG) #日志等级为DEBUG


# dir config
_C.data_dir = "glue_data/MRPC"

_C.flops_dataset_save_path =  "experiment/flopsdataset"

_C.output_dir = "experiment/MRPC/supermodeltrain/test32"

_C.predictor_save_path = "tmp.pt"

_C.save_flops_dataset_path = "experiment/generateflops"

_C.save_predictor_path =  "experiment/saved_predictor"

_C.cache_dir = ""

_C.overwrite_cache = False
_C.overwrite_output_dir = True

# other config
_C.server_ip = ""

_C.server_port = ""

_C.do_eval = True

_C.do_train = True

_C.fp16 = False

_C.fp16_opt_level = "01"

_C.local_rank = -1

_C.n_gpu = 1

_C.no_cuda = False

_C.save_steps = 51

_C.per_gpu_eval_batch_size = 8

_C.per_gpu_train_batch_size = 8

_C.eval_all_checkpoints = False

_C.evaluate_during_training = True

_C.seed = 42

_C.logging_steps = 50

_C.gradient_accumulation_steps = 1

# data config

_C.max_seq_length = 128

_C.tokenizer_name = ""

_C.do_lower_case = True

_C.task_name = "mrpc"

# train config

_C.output_mode = "classification"

_C.config_name = ""

_C.adam_epsilon = 1e-08

_C.num_train_epochs = 3.0

_C.max_steps = -1

_C.warmup_steps = 0

_C.weight_decay = 0.0

_C.learning_rate = 2e-05

_C.max_grad_norm = 1.0

# evo_config

_C.crossover_size = 50

_C.evo_iter = 30

_C.mutation_prob = 0.8

_C.mutation_size = 50

_C.flops_constraint = 10000000000

_C.flops_constraint_min = -1

_C.parent_size = 25

_C.population_size = 125

_C.search_model_path = ""

_C.constraint_model = 'flops'

_C.para_constraint = 66955008

_C.para_constraint_min = -1

_C.evo_log = "test"

# distill config

_C.alpha = 1.0

_C.beta = 100.0

_C.temperature = 5.0

# predictor config

_C.feature_norm = [7.002, 641.024, 1120.35264, 7.98982333]

_C.flops_norm = 6555246849

_C.flops_dataset_size = 10000

_C.predictor_train_steps = 2000

_C.hidden_dim = 400

_C.feature_dim = 4

_C.hidden_layer_num = 3

_C.predictor_bsz = 128

_C.predictor_lr = 1e-05

# attention dropout config

_C.attention_probs_dropout_prob = 0.5

_C.hidden_dropout_prob = 0.5

# model config

_C.model_type = "bert"

_C.student_model = "bert-base-uncased"

_C.student_sample_config = "default"

_C.model_name_or_path = "bert-base-uncased"

_C.teacher_model = "experiment/MRPC/default"

_C.teacher_model_sample_config = "default"

_C.super_model = "experiment/MRPC/supermodeltrain/test31"

_C.super_lr_schedule = "linear"

_C.teacher_sample_config = "default"

_C.geotype = 1


# train phase

_C.train_seq = ['head', 'ffn', 'layer']
_C.train_kind = ['all', 'ps', 'ps']

_C.head0_weight_decay = 0.0
_C.head0_learning_rate = 2e-5
_C.head0_num_train_epochs = 3.0
_C.head0_max_grad_norm = 1.0
_C.head0_super_lr_schedule = 'linear'
_C.head0_alpha = 1.0
_C.head0_beta = 100.0

_C.head1_weight_decay = 0.0
_C.head1_learning_rate = 2e-5
_C.head1_num_train_epochs = 3.0
_C.head1_max_grad_norm = 1.0
_C.head1_super_lr_schedule = 'linear'
_C.head1_alpha = 1.0
_C.head1_beta = 100.0


_C.ffn0_weight_decay = 0.0
_C.ffn0_learning_rate = 2e-5
_C.ffn0_num_train_epochs = 3.0
_C.ffn0_max_grad_norm = 1.0
_C.ffn0_super_lr_schedule = 'linear'
_C.ffn0_alpha = 1.0
_C.ffn0_beta = 100.0

_C.ffn1_weight_decay = 0.0
_C.ffn1_learning_rate = 2e-5
_C.ffn1_num_train_epochs = 3.0
_C.ffn1_max_grad_norm = 1.0
_C.ffn1_super_lr_schedule = 'linear'
_C.ffn1_alpha = 1.0
_C.ffn1_beta = 100.0

_C.ffn2_weight_decay = 0.0
_C.ffn2_learning_rate = 2e-5
_C.ffn2_num_train_epochs = 3.0
_C.ffn2_max_grad_norm = 1.0
_C.ffn2_super_lr_schedule = 'linear'
_C.ffn2_alpha = 1.0
_C.ffn2_beta = 100.0

_C.layer0_weight_decay = 0.0
_C.layer0_learning_rate = 2e-5
_C.layer0_num_train_epochs = 3.0
_C.layer0_max_grad_norm = 1.0
_C.layer0_super_lr_schedule = 'linear'
_C.layer0_alpha = 1.0
_C.layer0_beta = 100.0

_C.layer1_weight_decay = 0.0
_C.layer1_learning_rate = 2e-5
_C.layer1_num_train_epochs = 3.0
_C.layer1_max_grad_norm = 1.0
_C.layer1_super_lr_schedule = 'linear'
_C.layer1_alpha = 1.0
_C.layer1_beta = 100.0

_C.layer2_weight_decay = 0.0
_C.layer2_learning_rate = 2e-5
_C.layer2_num_train_epochs = 3.0
_C.layer2_max_grad_norm = 1.0
_C.layer2_super_lr_schedule = 'linear'
_C.layer2_alpha = 1.0
_C.layer2_beta = 100.0

_C.pown = 2.0

_C.rs_search_step = 1000.0

def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.output_dir, 'logger')
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options.
       How to use: python xx.py --cfg path_to_your_config.cfg test1 0 test2 True
       opts will return a list with ['test1', '0', 'test2', 'True'], yacs will compile to corresponding values
    """
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file",
                        help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


