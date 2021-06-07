from __future__ import absolute_import, division, print_function
import sys
sys.path.append('..')

import os
import os
import random
import json
from tqdm import tqdm, trange
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from torch import nn
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,AdamW, get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from utils.space_utils import (get_default_config,sample_subbert_config,get_represent_config, 
                                      sample_train_config, get_eval_config)
from utils.measure_utils import measure_flops, get_dummy_input
from utils.train_utils import load_and_cache_examples, evaluate, set_seed
import configs.config as config
from configs.config import args, logger
from model.supermodel import BertForSequenceClassification
from model.criterion import criterion_compute
from modules.probability import *
import pickle

max_info = 3.87


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

config.load_cfg_fom_args()

def train(args, train_dataset, teacher_model, super_model, d_criterion, tokenizer, device_using, train_type, train_kind, phase, distribution_optimizer, info):
    prob = 1
    if train_type == 'head':
        if train_kind == 'all':
            args.weight_decay = args.head0_weight_decay
            args.learning_rate = args.head0_learning_rate
            args.num_train_epochs = args.head0_num_train_epochs+args.head1_num_train_epochs
            args.max_grad_norm = args.head0_max_grad_norm
            args.super_lr_schedule = args.head0_super_lr_schedule
            args.alpha = args.head0_alpha
            args.beta = args.head0_beta
        else: 
            if phase == 0:
                args.weight_decay = args.head0_weight_decay
                args.learning_rate = args.head0_learning_rate
                args.num_train_epochs = args.head0_num_train_epochs
                args.max_grad_norm = args.head0_max_grad_norm
                args.super_lr_schedule = args.head0_super_lr_schedule
                args.alpha = args.head0_alpha
                args.beta = args.head0_beta
            else:
                args.weight_decay = args.head1_weight_decay
                args.learning_rate = args.head1_learning_rate
                args.num_train_epochs = args.head1_num_train_epochs
                args.max_grad_norm = args.head1_max_grad_norm
                args.super_lr_schedule = args.head1_super_lr_schedule
                args.alpha = args.head1_alpha
                args.beta = args.head1_beta
        
    elif train_type == 'ffn':
        if train_kind == 'all':
            args.weight_decay = args.ffn0_weight_decay
            args.learning_rate = args.ffn0_learning_rate
            args.num_train_epochs = args.ffn0_num_train_epochs+args.ffn1_num_train_epochs+args.ffn2_num_train_epochs
            args.max_grad_norm = args.ffn0_max_grad_norm
            args.super_lr_schedule = args.ffn0_super_lr_schedule
            args.alpha = args.ffn0_alpha
            args.beta = args.ffn0_beta
        else:
            if phase == 0:
                args.weight_decay = args.ffn0_weight_decay
                args.learning_rate = args.ffn0_learning_rate
                args.num_train_epochs = args.ffn0_num_train_epochs
                args.max_grad_norm = args.ffn0_max_grad_norm
                args.super_lr_schedule = args.ffn0_super_lr_schedule
                args.alpha = args.ffn0_alpha
                args.beta = args.ffn0_beta
            elif phase == 1:
                args.weight_decay = args.ffn1_weight_decay
                args.learning_rate = args.ffn1_learning_rate
                args.num_train_epochs = args.ffn1_num_train_epochs
                args.max_grad_norm = args.ffn1_max_grad_norm
                args.super_lr_schedule = args.ffn1_super_lr_schedule
                args.alpha = args.ffn1_alpha
                args.beta = args.ffn1_beta
            else:
                args.weight_decay = args.ffn2_weight_decay
                args.learning_rate = args.ffn2_learning_rate
                args.num_train_epochs = args.ffn2_num_train_epochs
                args.max_grad_norm = args.ffn2_max_grad_norm
                args.super_lr_schedule = args.ffn2_super_lr_schedule
                args.alpha = args.ffn2_alpha
                args.beta = args.ffn2_beta
            
    elif train_type == 'layer':
        if train_kind == 'all':
            args.weight_decay = args.layer0_weight_decay
            args.learning_rate = args.layer0_learning_rate
            args.num_train_epochs = args.layer0_num_train_epochs+args.layer1_num_train_epochs+args.layer2_num_train_epochs
            args.max_grad_norm = args.layer0_max_grad_norm
            args.super_lr_schedule = args.layer0_super_lr_schedule
            args.alpha = args.layer0_alpha
            args.beta = args.layer0_beta
        else:
            if phase == 0:
                args.weight_decay = args.layer0_weight_decay
                args.learning_rate = args.layer0_learning_rate
                args.num_train_epochs = args.layer0_num_train_epochs
                args.max_grad_norm = args.layer0_max_grad_norm
                args.super_lr_schedule = args.layer0_super_lr_schedule
                args.alpha = args.layer0_alpha
                args.beta = args.layer0_beta
            elif phase == 1:
                args.weight_decay = args.layer1_weight_decay
                args.learning_rate = args.layer1_learning_rate
                args.num_train_epochs = args.layer1_num_train_epochs
                args.max_grad_norm = args.layer1_max_grad_norm
                args.super_lr_schedule = args.layer1_super_lr_schedule
                args.alpha = args.layer1_alpha
                args.beta = args.layer1_beta
            else:
                args.weight_decay = args.layer2_weight_decay
                args.learning_rate = args.layer2_learning_rate
                args.num_train_epochs = args.layer2_num_train_epochs
                args.max_grad_norm = args.layer2_max_grad_norm
                args.super_lr_schedule = args.layer2_super_lr_schedule
                args.alpha = args.layer2_alpha
                args.beta = args.layer2_beta     
    else:
        pass
    

    if args.local_rank in [-1, 0]:
        path_tensorboard = os.path.join(args.output_dir, train_type+str(phase), 'tensorboard')
        tb_writer = SummaryWriter(path_tensorboard)


    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1 # t_total指总共更新更新model权重的次数
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = list(super_model.named_parameters()) + list(d_criterion.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
        
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.super_lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        super_model, optimizer = amp.initialize(super_model, optimizer, opt_level=args.fp16_opt_level)


    if args.n_gpu > 1:
        super_model = torch.nn.DataParallel(super_model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    if args.local_rank != -1:
        super_model = torch.nn.parallel.DistributedDataParallel(super_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    tr_loss = 0.0
    average_loss = 0.0
    train_avg_loss = 0.0
    soft_avg_loss = 0.0
    

    eval_config = get_eval_config(train_type, train_kind, phase)


    super_model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        dataset_length = len(epoch_iterator)
        update_prob_step = round(dataset_length*0.05)
        for step, batch in enumerate(epoch_iterator):
            super_model.train()
            teacher_model.eval()
            batch = tuple(t.to(device_using) for t in batch)
            inputs = {'input_ids':batch[0],
                      'attention_mask': batch[1],
                      'labels':batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  
            
            sample_config = sample_train_config(train_type, train_kind, phase, args.train_seq, distribution_optimizer, prob, reset_rand_seed = True)
           
            train_loss = 0
            soft_loss = 0
            distill_loss = 0     
            loss = 0
            for tem_config in sample_config:
                super_model.set_sample_config(tem_config)
                tem_train_loss, tem_soft_loss, tem_distill_loss = d_criterion(t_model=teacher_model,
                                                                s_model=super_model,
                                                                input_ids=inputs['input_ids'],
                                                                token_type_ids=inputs['token_type_ids'],
                                                                attention_mask=inputs['attention_mask'],
                                                                labels=inputs['labels'],
                                                                args=args)

                tem_loss = args.alpha * tem_train_loss + (1 - args.alpha) * tem_soft_loss + args.beta * tem_distill_loss  

                train_loss += tem_train_loss
                soft_loss += tem_soft_loss
                distill_loss +=tem_distill_loss     
                loss += tem_loss

            if args.n_gpu > 1:
                loss = loss.mean()  
                train_loss = train_loss.mean()
                soft_loss = soft_loss.mean()
                distill_loss = distill_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                train_loss = train_loss / args.gradient_accumulation_steps
                soft_loss = soft_loss / args.gradient_accumulation_steps
                distill_loss = distill_loss / args.gradient_accumulation_step
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(super_model.parameters(), args.max_grad_norm)

       
            record_information = -train_loss

        
            sample_config_one_hot = config2encode(sample_config[0])
            distribution_optimizer.record_information(sample_config_one_hot, record_information)
            distribution_optimizer.update()
            
            
            
            tr_loss += loss.item()
            average_loss += loss.item()
            train_avg_loss += train_loss.item()
            soft_avg_loss += soft_loss.item()
            if (step + 1) % update_prob_step == 0:
                prob = distribution_optimizer.p_model.get_infoentropy()/max_info 
                info.append(prob)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(super_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  
                        current_best_index = np.argmax(distribution_optimizer.p_model.theta, axis=1)
                        current_best = encode2config(index_to_one_hot(current_best_index, 12))
                        super_model.set_sample_config(current_best)
                        result_random,loss = evaluate(args, super_model, tokenizer, device_using)
                        if args.task_name != "mnli":
                            acc = result_random['acc']
                            tb_writer.add_scalar('acc:', acc, global_step)
                            tb_writer.add_scalar('loss:', loss, global_step)
                        else:
                            mnli_acc = result_random["mnli/acc"]
                            mnli_mmacc = result_random["mnli-mm/acc"]
                            tb_writer.add_scalar('mnli_acc:', mnli_acc, global_step)
                            tb_writer.add_scalar('mnli-mm_acc:', mnli_mmacc, global_step)
                            tb_writer.add_scalar('loss:', loss, global_step)
        
                    tb_writer.add_scalar('all_lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('all total loss', average_loss / args.logging_steps, global_step)
                    tb_writer.add_scalar('all train loss', train_avg_loss / args.logging_steps, global_step)
                    tb_writer.add_scalar('all soft loss', soft_avg_loss / args.logging_steps, global_step)

                    average_loss = 0.0
                    train_avg_loss = 0.0
                    soft_avg_loss = 0.0

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            
    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

        output_dir = os.path.join(args.output_dir, train_type+str(phase))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
     
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step



def main():

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

 
    if args.server_ip and args.server_port:
 
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

 
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    device_using = device


    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)


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
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.teacher_model,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)


    teacher_model_config = config_class.from_pretrained(args.config_name if args.config_name else args.teacher_model,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    teacher_model_config.output_hidden_states = True
    teacher_model = model_class.from_pretrained(args.teacher_model,
                                        from_tf=bool('.ckpt' in args.teacher_model),
                                        config=teacher_model_config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

 
    super_model_config = config_class.from_pretrained(args.config_name if args.config_name else args.super_model,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    super_model_config.output_hidden_states = True
    super_model_config.hidden_dropout_prob = args.hidden_dropout_prob
    super_model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    
    super_model = model_class.from_pretrained(args.super_model,
                                        from_tf=bool('.ckpt' in args.super_model),
                                        config=super_model_config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    
 
    f=open("/arc/arc.json","r")
    data=list()
    for line in f:
        data.append(json.loads(line))
    f.close()


    super_model_sample_config =  get_default_config(super_model_config)
    if args.teacher_model_sample_config == 'default':
        teacher_model_sample_config = get_default_config(teacher_model_config)
    else:
        teacher_model_sample_config = data[-1]    


    d_criterion = criterion_compute(teacher_model_sample_config, super_model_sample_config)

    if args.local_rank == 0:
        torch.distributed.barrier()  


    teacher_model.to(device_using)
    teacher_model.set_sample_config(teacher_model_sample_config)
    super_model.to(device_using)
    super_model.set_sample_config(super_model_sample_config)
    d_criterion.to(device_using)
    

    cate = [4]+[12]*12 
    distribution_optimizer = SNG(cate)


  
    if args.local_rank != -1:
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
        super_model = torch.nn.parallel.DistributedDataParallel(super_model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
    elif args.n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
        super_model = torch.nn.DataParallel(super_model)


    logger.info("Training/evaluation parameters %s", args)
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    

    info = []

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        for train_type, train_kind in zip(args.train_seq, args.train_kind):
            if train_type == 'head':
                if train_kind == 'all':
                    train_phase = 1
                else:
                    train_phase = 2
            elif train_type == 'ffn':
                if train_type == 'all':
                    train_phase =1
                else:
                    train_phase = 3
            elif train_type == 'layer':
                if train_type == 'all':
                    train_phase = 1
                else:
                    train_phase = 3
            for phase in range(train_phase):
                global_step, tr_loss = train(args, train_dataset, teacher_model, super_model, d_criterion, tokenizer, device_using, train_type, train_kind, phase, distribution_optimizer,info)
                logger.info(train_type+str(train_phase)+":global_step = %s, average loss = %s", global_step, tr_loss)
                print("info",info)
  
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)

        model_to_save = super_model.module if hasattr(super_model,
                                                'module') else super_model  
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

  
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

 
        super_model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        super_model.to(device_using)

    current_best_index = np.argmax(distribution_optimizer.p_model.theta, axis=1)
    current_best = encode2config(index_to_one_hot(current_best_index, 12))
    super_model.set_sample_config(current_best)
    result,loss = evaluate(args, super_model, tokenizer, device_using)    
    save_file = os.path.join(args.output_dir, 'distribution.pkl')
    with open(save_file, 'wb') as f:
        picklestring = pickle.dump(distribution_optimizer, f)
if __name__ == "__main__":
    main()