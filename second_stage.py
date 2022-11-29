# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Drawing & fine-tuning stage.

"""


import dataclasses
import logging
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

import ipdb
import copy
import math

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


from trainer import Trainer
from training_args import TrainingArguments

from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertLayer
from pruning_utils import *
from utils import get_pruning_mask,set_logging_config

logger = None

os.environ['TFHUB_CACHE_DIR'] = '/root/tfhub_modules'

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default=None
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
@dataclass
class MyArguments:
    """
    some extra args
    """
    dataset_name: Optional[str] = field(
        default="glue", metadata={"help": "dataset name"}
    )
    num_labels: Optional[int] = field(
        default=2, metadata={"help": "num of labels"}
    )
    num_examples: Optional[int] = field(
        default=1000, metadata={"help": "num of attack examples"}
    )
    attack_every_epoch: bool = field(default=False, metadata={"help": "Whether to attack every epoch."})
    attack_all: bool = field(default=False, metadata={"help": "Whether to attack with 3 methods."})
    neighbour_vocab_size: Optional[int] = field(default=10,metadata={"help": "only bert_attack uses this"})
    modify_ratio: Optional[float] = field(default=0.15,metadata={"help": "only bert_attack uses this"})
    sentence_similarity: Optional[float] = field(default=0.85,metadata={"help": "only bert_attack uses this"})
    eval_after_train: bool = field(default=True)
    not_attack: bool = field(default=False)
    acc_threshold: float = field(default=0.6)
    layers_pruned_file: Optional[str] = field(default=None)
    random_reinit: bool = field(default=False)
    without_struct: bool = field(default=False)
    epsilon: float = field(default=1.0)
    attack_method: str = field(default="textfooler")
    attack_seed: int = field(default=42)
# attack test
import argparse
import os
import csv
import logging
import textattack
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AdamW
)
from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
logger = logging.getLogger(__name__)
import torch


from attack_utils import *


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,MyArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, my_args = parser.parse_args_into_dataclasses()
    # dataset = HuggingFaceDataset(my_args.dataset_name, subset="sst2" if data_args.task_name=="sst-2" else data_args.task_name, split=data_args.valid)

    # Set up handler for logging
    set_logging_config(training_args.output_dir)
    global logger
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger(__name__)
    logger.info("Running Draw and Retrain Stage!!!!!!!")
    # make sure that we will not perform the pruning twice
    assert training_args.prune_before_train and (not training_args.prune_before_eval)
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        if my_args.dataset_name=="imdb":
            num_labels = 2
            output_mode= "classification"
        elif my_args.dataset_name=="ag_news":
            num_labels = 4
            output_mode= "classification"
        else: # my_args.dataset_name=="glue":
            num_labels = glue_tasks_num_labels[data_args.task_name]
            output_mode = glue_output_modes[data_args.task_name]

    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # do_lower_case=True # todo modified
    )

    if training_args.load_from_pruned:
        new_config = copy.deepcopy(config)
        new_config.self_pruning_ratio = training_args.self_pruning_ratio
        new_config.inter_pruning_ratio = training_args.inter_pruning_ratio
    else:
        new_config = config
    logger.info("model_name used in prune and retrain stage :"+model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=new_config,
        cache_dir=model_args.cache_dir,
    )
    # reinit modules
    if my_args.random_reinit:
        def init_weights(config, module):
            """Initialize the weights"""
            if isinstance(module, torch.nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        for layer_index in range(12):
            # query
            module = model.bert.encoder.layer[layer_index].attention.self.query
            init_weights(new_config,module)
            # key
            module = model.bert.encoder.layer[layer_index].attention.self.key
            init_weights(new_config, module)
            # value
            module = model.bert.encoder.layer[layer_index].attention.self.value
            init_weights(new_config, module)

            # attention output dense
            module = model.bert.encoder.layer[layer_index].attention.output.dense
            init_weights(new_config, module)

            # intermediate dense
            module = model.bert.encoder.layer[layer_index].intermediate.dense
            init_weights(new_config, module)

            # output dense
            module = model.bert.encoder.layer[layer_index].output.dense
            init_weights(new_config, module)
        # pooler dense
        module = model.bert.pooler.dense
        init_weights(new_config, module)

    import utils
    from torch.utils.data import DataLoader
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    # train_dataset = None
    # train_loader = None
    # training_args.train_batch_size = 32
    if my_args.dataset_name == 'imdb' or my_args.dataset_name == 'ag_news':
        data_args.task_name=None
        data_args.valid = "test"
    elif data_args.task_name == "mnli":
        data_args.valid = "validation_matched"
    else:
        data_args.valid = "validation"
    train_dataset = utils.Huggingface_dataset(data_args, tokenizer, name_or_dataset="glue" if my_args.dataset_name is None else my_args.dataset_name, subset="sst2" if data_args.task_name=="sst-2" else data_args.task_name,split="train")
    train_loader = DataLoader(train_dataset, batch_size=training_args.train_batch_size, shuffle=True, collate_fn=collator)

    eval_dataset = utils.Huggingface_dataset(data_args, tokenizer, name_or_dataset="glue" if my_args.dataset_name is None else my_args.dataset_name, subset="sst2" if data_args.task_name=="sst-2" else data_args.task_name
                                             , split=data_args.valid)
    eval_loader = DataLoader(eval_dataset, batch_size=training_args.train_batch_size, shuffle=False, collate_fn=collator)

    test_dataset = None
    # test_dataset = utils.Huggingface_dataset(data_args, tokenizer, name_or_dataset="glue", subset="sst2" if data_args.task_name=="sst-2" else data_args.task_name
    #                                          , split="test")

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Calculate at which step we draw the ticket. We provide several ways to
    # determine this depending on the value of the argument `args.slimming_coef_step`
    # Let t = args.slimming_coef_step (float type)
    #
    #                       /-- int(t), if t >= 1
    # step to draw ticket =  -- round(t * num_steps_per_epoch), if 0 < t < 1
    #                       \-- floor(t), if t <= 0, used as random seed for random pruning exp
    #
    # If t >= 1, we will just use `t` as the step index and draw the ticket at that step.
    # If 0 < t < 1, `t` represents that we draw the ticket after (t*100)% of the
    # total number of steps in the first epoch.
    # If t <= 0, we apply random pruning and use `-floor(t)` as the random seed for pruning.
    if training_args.slimming_coef_step >= 1.0:
        training_args.slimming_coef_step = int(training_args.slimming_coef_step)
    elif training_args.slimming_coef_step > 0.0:
        num_steps_per_epoch = len(train_dataset) / training_args.train_batch_size
        training_args.slimming_coef_step = round(training_args.slimming_coef_step * num_steps_per_epoch)
    else:
        training_args.slimming_coef_step = math.floor(training_args.slimming_coef_step)

    # Pruning
    if my_args.epsilon!=1.0 and training_args.slimming_coef_step>0:
        len_dataset = 67349
        if my_args.dataset_name=="imdb":
            len_dataset = 25000
            per_device_batch_size = 8
        elif my_args.dataset_name == "ag_news":
            len_dataset = 120000
            per_device_batch_size = 8
        elif my_args.dataset_name=="glue":
            if data_args.task_name=="sst2" or data_args.task_name=="sst-2":
                len_dataset = 67349
                per_device_batch_size = 32
            elif data_args.task_name=="qnli":
                len_dataset = 104743
                per_device_batch_size = 32
            elif data_args.task_name=="qqp":
                len_dataset = 363846
                per_device_batch_size = 32
            elif data_args.task_name=="mnli":
                len_dataset = 392702
                per_device_batch_size = 32


        all_steps = int(len_dataset * 1 // per_device_batch_size)
        self_slimming_coefs = np.load(training_args.self_slimming_coef_file)
        inter_slimming_coefs = np.load(training_args.inter_slimming_coef_file)



# if training_args.do_train:
    pruning(logger, model, my_args, new_config, training_args)

    # training!
    train_outputs = trainer.train(train_dataloader = train_loader, eval_dataloader=eval_loader,
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,
        config=config, l1_loss_coef=0.0, lottery_ticket_training=True
    )
    # save model
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)


    # eval after training!
    eval_acc = 0.0
    if my_args.eval_after_train:
        from tqdm import tqdm
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('Evaluating...')
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(eval_loader)
            for model_inputs, labels in pbar:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                # logits = model(**model_inputs).logits
                logits = model(**model_inputs, return_dict=False)[0]
                _, preds = logits.max(dim=-1)  # todo 报错
                correct += (preds == labels.squeeze(-1)).sum().item()
                total += labels.size(0)
                pbar.set_description("correct: {}, total: {}, acc: {} ".format(correct,total,correct / (total + 1e-13)))
            accuracy = correct / (total + 1e-13)
        logger.info(f'Accuracy: {accuracy}')
        eval_acc = accuracy
    # Evaluation

    eval_results = {}

    logger.info(f'All Accuracy: {accuracy}')

    # do attack
    if eval_acc>=my_args.acc_threshold and not my_args.not_attack:
        logger.info("Attacking after training!")
        training_args.results_file = 'attack_log.csv'
        training_args.task_name = data_args.task_name
        training_args.num_examples = my_args.num_examples # 1000
        training_args.seed = 42
        training_args.save_perturbed = 0
        training_args.perturbed_file = "bert_textfooler.csv"
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        if my_args.attack_all:

            attack_methods = ["textfooler","textbugger","bertattack"] #todo done bertattack放最后一个，因为要改变攻击的参数！！！
        else:
            attack_methods = [my_args.attack_method]
        for attack_method in attack_methods:
            logger.info("attack method is {}".format(attack_method))
            if attack_method=="bertattack":
                my_args.neighbour_vocab_size = 50
                my_args.modeify_ratio = 0.9
                my_args.sentence_similarity = 0.2
                attack = build_weak_attacker(my_args, model_wrapper,attack_method)
            else:
                attack = build_english_attacker(my_args, model_wrapper,attack_method)
            dataset = HuggingFaceDataset(my_args.dataset_name, subset="sst2" if data_args.task_name=="sst-2" else data_args.task_name, split=data_args.valid)

            # for attack
            attack_args = textattack.AttackArgs(num_examples=training_args.num_examples,
                                            disable_stdout=True, random_seed=my_args.attack_seed)

            attacker = textattack.Attacker(attack, dataset, attack_args)
            num_results = 0
            num_successes = 0
            num_failures = 0


            if training_args.save_perturbed:
                with open(training_args.perturbed_file, 'w', encoding='utf-8', newline="") as f:
                    csv_writer = csv.writer(f, delimiter='\t')
                    csv_writer.writerow(['sentence', 'label'])
                    f.close()
            # attacking
            for result in attacker.attack_dataset():
                # logger.info(result)
                num_results += 1
                if (
                        type(result) == SuccessfulAttackResult
                        or type(result) == MaximizedAttackResult
                ):
                    num_successes += 1
                if type(result) == FailedAttackResult:
                    num_failures += 1

                if training_args.save_perturbed:
                    with open(training_args.perturbed_file, 'a', encoding='utf-8', newline="") as f:
                        csv_writer = csv.writer(f, delimiter='\t')
                        csv_writer.writerow([result.perturbed_result.attacked_text.text, result.perturbed_result.ground_truth_output])

            logger.info("[Succeeded / Failed / Total] {} / {} / {}".format(num_successes, num_failures, num_results))

            # compute metric
            original_accuracy = (num_successes + num_failures) * 100.0 / num_results
            accuracy_under_attack = num_failures * 100.0 / num_results
            attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
            # out_csv = open(training_args.results_file, 'a', encoding='utf-8', newline="")
            # csv_writer = csv.writer(out_csv)
            # csv_writer.writerow([training_args.model_name_or_path, original_accuracy, accuracy_under_attack, attack_succ])
            # out_csv.close()
            logger.info("[Accuracy / Aua / Attack_success] {} / {} / {}".format(original_accuracy, accuracy_under_attack, attack_succ))


    return eval_results


def choose_slimming_step(all_steps, inter_slimming_coefs, logger, model, my_args, self_slimming_coefs, training_args):
    import pandas as pd
    inter_values = pd.DataFrame(np.zeros([20, 20]))
    self_values = pd.DataFrame(np.zeros([20, 20]))
    for gap1 in range(0, 20, 1):
        for gap2 in range(gap1, 20, 1):
            step1 = all_steps * gap1 // 20
            step2 = all_steps * gap2 // 20

            if training_args.self_pruning_method != 'layerwise':
                self_dis, self_norm_dis = cal_mask_distance_in_self_heads_of_two_step_prune_heads_global(model=model,
                                                                                                         self_slimming_coef_records=self_slimming_coefs,
                                                                                                         slimming_step1=step1,
                                                                                                         slimming_step2=step2,
                                                                                                         self_pruning_method=training_args.self_pruning_method,
                                                                                                         self_pruning_ratio=training_args.self_pruning_ratio)
            else:
                self_dis = 0
                self_norm_dis = 0
            inter_dis, inter_norm_dis = cal_mask_distance_in_inter_neurons_of_two_step(model=model,
                                                                                       inter_slimming_coef_records=inter_slimming_coefs,
                                                                                       slimming_step1=step1,
                                                                                       slimming_step2=step2,
                                                                                       inter_pruning_method=training_args.inter_pruning_method,
                                                                                       inter_pruning_ratio=training_args.inter_pruning_ratio)

            self_values[gap1][gap2] = self_norm_dis
            inter_values[gap1][gap2] = inter_norm_dis
    epsilon = my_args.epsilon
    for gap in range(0, 15, 1):
        if self_values[gap][gap + 1] < epsilon and self_values[gap][gap + 2] < epsilon and self_values[gap][
            gap + 3] < epsilon and self_values[gap][gap + 4] < epsilon and self_values[gap][gap + 5] < epsilon and \
                inter_values[gap][gap + 1] < epsilon and inter_values[gap][gap + 2] < epsilon and inter_values[gap][
            gap + 3] < epsilon and inter_values[gap][gap + 4] < epsilon and inter_values[gap][gap + 5] < epsilon:
            training_args.slimming_coef_step = all_steps * gap // 20
            logger.info(
                "New slimming_coef_step is : {}, new slimming epoch is: {}".format(training_args.slimming_coef_step,
                                                                                   float(gap) / 20))
            break


def pruning(logger, model, my_args, new_config, training_args):
    # select layers to be pruned
    if my_args.layers_pruned_file == None:
        layers_pruned = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    else:
        with open(my_args.layers_pruned_file, 'r') as layers_pruned_file:
            str_list = layers_pruned_file.readline().split(',')
            layers_pruned = [int(n) for n in str_list]
    # Prune intermediate neurons in FFN modules based on the learnable coefficients
    bert_layers = []
    for m in model.modules():
        # print(m)
        if isinstance(m, BertLayer):
            bert_layers.append(m)
    # Get the coefficients for pruning, which has shape (num_hidden_layers, num_inter_neurons)
    if training_args.slimming_coef_step > 0:
        slimming_coefs = np.load(training_args.inter_slimming_coef_file)[:, training_args.slimming_coef_step - 1, :]

    else:
        # Random pruning
        # Get internal state of the random generator first
        rand_state = np.random.get_state()
        # Set random seed
        np.random.seed(-training_args.slimming_coef_step)
        slimming_coefs = np.random.rand(
            len(bert_layers), bert_layers[0].intermediate.dense.out_features)
        # Reset internal state
        np.random.set_state(rand_state)
    # If we do layerwise pruning, calculate the threshold along the last dimension
    # of `slimming_coefs`, which corresponds to the self-attention heads in each layer;
    # otherwise, calculate the threshold along all dimensions in `slimming_coefs`.
    quantile_axis = -1 if training_args.inter_pruning_method == 'layerwise' else None
    if training_args.inter_pruning_method != 'layerwise':
        for r in range(slimming_coefs.shape[0]):
            if not r in layers_pruned:
                for c in range(slimming_coefs[r].shape[0]):
                    slimming_coefs[r][c] = float("inf")
    threshold = np.quantile(slimming_coefs, training_args.inter_pruning_ratio, axis=quantile_axis, keepdims=True)
    layers_masks = slimming_coefs > threshold
    # if my_args.without_struct:
    #     for m, mask in zip(bert_layers, layers_masks):
    #         pruned_inter_neurons = [i for i in range(new_config.intermediate_size) if mask[i] == 0]
    #         m.without_struct_reinit_discarded_neurons(pruned_inter_neurons,config=new_config,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # else:
    for m, mask in zip(bert_layers, layers_masks):
        pruned_inter_neurons = [i for i in range(new_config.intermediate_size) if mask[i] == 0]
        logger.info('{} neurons are pruned'.format(len(pruned_inter_neurons)))
        m.prune_inter_neurons(pruned_inter_neurons)
    # Prune self-attention heads based on the learnable coefficients
    attention_modules = []
    # slimming_coefs = []
    for m in model.modules():
        if isinstance(m, BertAttention):
            attention_modules.append(m)
    # Get the coefficients for pruning, which has shape (num_hidden_layers, num_attention_heads)
    if training_args.slimming_coef_step > 0:
        slimming_coefs = np.load(training_args.self_slimming_coef_file)[:, training_args.slimming_coef_step - 1, :]
    else:
        # random pruning
        # get internal state of the random generator first
        rand_state = np.random.get_state()
        # set random seed
        np.random.seed(-training_args.slimming_coef_step)
        slimming_coefs = np.random.rand(len(attention_modules), new_config.num_attention_heads)
        # reset internal state
        np.random.set_state(rand_state)
    # If we do layerwise pruning, calculate the threshold along the last dimension
    # of `slimming_coefs`, which corresponds to the self-attention heads in each layer;
    # otherwise, calculate the threshold along all dimensions in `slimming_coefs`.

    if training_args.inter_pruning_method != 'layerwise':
        for r in range(slimming_coefs.shape[0]):
            if not r in layers_pruned:
                for c in range(slimming_coefs[r].shape[0]):
                    slimming_coefs[r][c] = float("inf")
    quantile_axis = -1 if training_args.self_pruning_method == 'layerwise' else None
    threshold = np.quantile(slimming_coefs, training_args.self_pruning_ratio, axis=quantile_axis,
                            keepdims=True)

    # layers_masks = slimming_coefs > threshold
    def is_every_layer_one_head_survived(slimming_coefs, quantile_axis, threshold):
        layers_masks = slimming_coefs > threshold
        for idx in range(len(layers_masks)):
            mask = layers_masks[idx]
            if sum([1 if i == True else 0 for i in mask]) == 0:
                return idx
        return -1

    if training_args.self_pruning_method != 'layerwise':
        while True:
            idx = is_every_layer_one_head_survived(slimming_coefs, quantile_axis, threshold)
            if idx != -1:
                p = list(slimming_coefs[idx]).index(max(slimming_coefs[idx]))
                slimming_coefs[idx][p] = float('inf')
                quantile_axis = -1 if training_args.self_pruning_method == 'layerwise' else None
                threshold = np.quantile(slimming_coefs, training_args.self_pruning_ratio, axis=quantile_axis,
                                        keepdims=True)
            else:
                break
    # final masks
    layers_masks = slimming_coefs > threshold
    if my_args.without_struct:
        for m, mask in zip(attention_modules, layers_masks):
            pruned_heads = [i for i in range(len(mask)) if mask[i] == 0]
            m.without_struct_reinit_discarded_heads(pruned_heads, config=new_config,
                                                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        for m, mask in zip(attention_modules, layers_masks):
            pruned_heads = [i for i in range(len(mask)) if mask[i] == 0]  # 计算出当前self-attention要剪掉哪些head
            print('pruned heads: {}'.format(str(pruned_heads)))
            m.prune_heads(pruned_heads)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
