import sys
sys.path.append('..')
from utils.train_utils import load_and_cache_examples, set_seed
from app.train_predictor import FlopsPredictor
from model.supermodel import BertForSequenceClassification
from utils.space_utils import get_default_config, get_subbert_search_space, get_represent_config
from utils.measure_utils import measure_flops, get_dummy_input
from transformers import glue_compute_metrics as compute_metrics
from transformers import BertConfig, BertTokenizer
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                          TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import pdb
import json
import os
from tqdm import tqdm, trange
import glob
import argparse
from configs.config import args, get_logger
import configs.config as config
import logging
import pickle
from utils.space_utils import *
import math
config.load_cfg_fom_args()
save_path = os.path.join("evolog", args.task_name)

if not os.path.exists(save_path):
    os.mkdir(save_path)
    
logger = get_logger(os.path.join(save_path, args.evo_log+".log"))
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

max_flops = 22353349888
max_para = 109482240
    
    

class randomsearch(object):

    def __init__(self, args, super_model, tokenizer, device_using):
        self.args = args
        self.flops_constraint = self.args.flops_constraint
        self.bert_config = None
        self.device_using = device_using
        self.super_model = super_model
        self.tokenizer = tokenizer


    def satisfy_constraints(self, config):
        satisfy = True
        if args.constraint_model == 'flops':
            dummy_input = get_dummy_input(self.args, self.device_using)
            flops = measure_flops(self.super_model, config, dummy_input)
            if flops > self.flops_constraint:
                satisfy = False
        else:
            para_num = self.super_model.get_sampled_params_numel(config)
            if para_num > self.args.para_constraint:
                satisfy =False
        return satisfy
    def get_latency(self, config):
        if args.constraint_model == 'flops':
            dummy_input = get_dummy_input(self.args, self.device_using)
            result = measure_flops(self.super_model, config, dummy_input)
        else:
            result = self.super_model.get_sampled_params_numel(config)
        return result 
        
    def getreward(self, config, acc, constraints):
        latency = self.get_latency(config)
        if latency < constraints:
            reward = acc
        else:
            if args.constraint_model == 'flops':
                rate = (latency - constraints)/ (max_flops - constraints)
            else:
                rate = (latency - constraints) / (max_para - constraints)
            reward = acc * math.pow((1-rate), args.pown)
        return reward

    def validate_all(self, configs):

        eval_task_names = (
            "mnli", "mnli-mm") if self.args.task_name == "mnli" else (self.args.task_name,)  # mnli.mm
        for eval_task in eval_task_names:
            eval_dataset = load_and_cache_examples(
                self.args, eval_task, self.tokenizer, evaluate=True)

            self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * \
                max(1, self.args.n_gpu)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

            if self.args.n_gpu > 1:
                model = torch.nn.DataParallel(self.model)

            self.super_model.set_sample_config(configs)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):

                self.super_model.eval()
                batch = tuple(t.to(self.device_using) for t in batch)
                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'labels': batch[3]}
                    if self.args.model_type != 'distilbert':
                        inputs['token_type_ids'] = batch[2] if self.args.model_type in ['bert',
                                                                                        'xlnet'] else None  
                    outputs = self.super_model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()

                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(
                        preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            if self.args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif self.args.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)  
            if self.args.task_name == "cola":
                acc = result['mcc']
            elif self.args.task_name == "mrpc":
                acc = result['acc_and_f1']
            # mnli最后再跑
            elif self.args.task_name == "mnli" and eval_task == "mnli-mm":
                acc = (result['mnli-mm/acc'] + acc)/2
            elif self.args.task_name == "mnli" and eval_task == "mnli":
                acc = result['mnli/acc']
            else:
                acc = result['acc']
        return acc


    def run_search(self):

        if self.args.constraint_model == 'flops':
            constraints = self.flops_constraint
        else:
            constraints = self.args.para_constraint
        

        file_path = os.path.join(args.search_model_path, "distribution.pkl")
        with open(file_path, 'rb') as f:
            optimizer = pickle.load(f)
        

        best_acc = 0
        best_arc = -1
        distribution_best_acc = []
        distribution_best_constraints= []
        best_arc_acc=[]

        for i in range(int(args.rs_search_step)):
            logger.info(f"| Start Iteration {i}:")
            sample_one_hot = optimizer.sampling()
            sample_config = encode2config(sample_one_hot)
            if self.args.constraint_model == 'flops':
                dummy_input = get_dummy_input(self.args, self.device_using)
                constraints_measure = measure_flops(self.super_model, sample_config, dummy_input)
            else:
                constraints_measure = self.super_model.get_sampled_params_numel(sample_config)      
            acc=self.validate_all(sample_config)   

            current_best_index = np.argmax(optimizer.p_model.theta, axis=1)
            current_best = encode2config(index_to_one_hot(current_best_index, 12))
            if self.args.constraint_model == 'flops':
                dummy_input = get_dummy_input(self.args, self.device_using)
                constraints_measure_current_best = measure_flops(self.super_model, current_best, dummy_input)
            else:
                constraints_measure_current_best = self.super_model.get_sampled_params_numel(current_best) 
            distribution_best_constraints.append(constraints_measure_current_best)
            distribution_bestsample_acc = self.validate_all(current_best)
            distribution_best_acc.append(distribution_bestsample_acc)

            reward = self.getreward(sample_config, acc, constraints)
            optimizer.record_information(sample_one_hot, reward)
            optimizer.update()

            if acc > best_acc and constraints_measure <constraints:
                logger.info(f"| current_best_acc:{best_acc}")
                best_acc = acc
                best_arc = sample_config
                best_arc_acc.append(best_acc)
        best_arc_acc = np.array(best_arc_acc)
        size_best_arc_acc = best_arc_acc.size
        x_size_best_arc_acc  = np.arange(size_best_arc_acc)
        plt.plot(x_size_best_arc_acc,best_arc_acc)
        plt.savefig(os.path.join(save_path, args.evo_log+".png"))
   
        return best_arc, best_acc






def main():
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else: 
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    device_using = device

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.search_model_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.output_hidden_states = True
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.search_model_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    super_model = model_class.from_pretrained(args.search_model_path,
                                              from_tf=bool(
                                                  '.ckpt' in args.search_model_path),
                                              config=config,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
    if args.local_rank == 0:
        torch.distributed.barrier()
    super_model.to(device_using)
    arc_config = get_default_config(config)
    super_model.set_sample_config(arc_config)
    evolver = randomsearch(args, super_model, tokenizer, device_using)
    best_config, best_acc = evolver.run_search()

    logger.info("************************************************************************")
    logger.info(f"best accuracy : {best_acc}")
    logger.info(f"best config : {best_config}")
    dummy_input = get_dummy_input(args, device_using)
    flops = measure_flops(super_model, best_config, dummy_input)
    logger.info(f"| flops: {flops}")
    para_num = super_model.get_sampled_params_numel(best_config)
    logger.info(f"| para: {para_num}")
    logger.info("************************************************************************")

if __name__ == "__main__":
    main()
