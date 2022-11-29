"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)

import utils as utils
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# For recording the learnable coefficients for self-attention heads and
self_slimming_coef_records = None
inter_slimming_coef_records = None


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default='sst2', type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('/root/robust_transfer/saved_models/'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=False)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # hyper-parameters
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=float, default=0.2,help="num_train_epochs")
    parser.add_argument('--lr', type=float, default=2e-5,help="search learning rate")
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_correction', default=True)
    parser.add_argument('-f', '--not_force_overwrite',action="store_true") # 只有传入了这个参数才会是true
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir',type=str,default='/root/Early_Robust/saved_models/')

    # Adversarial training specific
    parser.add_argument('--adv_steps', default=5, type=int,
                        help='Number of gradient ascent steps for the adversary')
    parser.add_argument('--adv_lr', default=0.01, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0.05, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')

    # added early ticket related params
    parser.add_argument('--save_steps', default=2500, type=int, help='')
    parser.add_argument('--max_seq_length', default=128, type=int, help='')
    parser.add_argument('--l1_loss_self_coef', default=1e-4, type=float, help='')
    parser.add_argument('--l1_loss_inter_coef', default=1e-4, type=float, help='')
    parser.add_argument('--l1_loss_coef', default=1e-4, type=float, help='')
    parser.add_argument('--max_epochs', default=1, type=int, help='')

    parser.add_argument('--cal_time', action="store_true")

    args = parser.parse_args()
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'
    return args


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main(args):

    from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(log_dir="./runs/coef_grad_norm",flush_secs=60)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif args.not_force_overwrite:
        print("skip stage 1 ")
        return

    print("search stage output_dir:"+str(output_dir))
    log_file = os.path.join(output_dir, 'INFO.log')
    logger.addHandler(logging.FileHandler(log_file))
    logger.info("Running Search Stage!!!!!!!")
    # pre-trained config tokenizer model
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset_name=="ag_news":
        args.num_labels = 4
    elif args.task_name=="mnli":
        args.num_labels=3

    # config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels,mirror='tuna')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)

    # Perform the searching stage in ER for both self-attention heads and
    # intermediate neurons in two-layer FFN modules.
    config.self_slimming = True
    config.inter_slimming =  True
    # Initialize the list for recording the learnable coefficients in ER.
    # Separately record the coefficients in different layers.
    global self_slimming_coef_records, inter_slimming_coef_records
    self_slimming_coef_records = [[] for _ in range(config.num_hidden_layers)]
    inter_slimming_coef_records = [[] for _ in range(config.num_hidden_layers)]

    # todo modified
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    # model = modeling_utils.PreTrainedModel.from_pretrained(args.model_name,config=config) # 直接调用自己写的modeling utils
    model.to(device)

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    # for training
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        args.task_name=None
        args.valid = "test"
    elif args.task_name=="mnli":
        args.valid = "validation_matched"

    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator) # todo
    # train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
    logger.info("train dataset length: "+ str(len(train_dataset)))
    # for dev
    dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                            subset=args.task_name, split=args.valid)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # for test
    if args.do_test:
        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )

    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    global_step = 0
    steps_trained_in_current_epoch = 0

    # todo Save model checkpoint at initialization
    output_dir = os.path.join(args.output_dir, "checkpoint-0")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    # 把output_dir放回去
    output_dir = args.output_dir
    # adversarial training
    try:
        import time

        best_accuracy = 0
        best_dev_epoch = 0
        while True:
            if global_step >= num_training_steps:
                break
            logger.info('Training...')
            model.train()
            avg_loss = utils.ExponentialMovingAverage()
            pbar = tqdm(train_loader)
            for model_inputs, labels in pbar:
                epoch = global_step//(len(train_dataset)//args.bsz)
                if global_step >= num_training_steps:
                    break
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                model.zero_grad()
                # for freelb
                word_embedding_layer = model.get_input_embeddings()
                input_ids = model_inputs['input_ids']
                attention_mask = model_inputs['attention_mask']
                embedding_init = word_embedding_layer(input_ids)
                # initialize delta
                if args.adv_init_mag > 0:
                    input_mask = attention_mask.to(embedding_init)
                    input_lengths = torch.sum(input_mask, 1)
                    if args.adv_norm_type == 'l2':
                        delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                        dims = input_lengths * embedding_init.size(-1)
                        magnitude = args.adv_init_mag / torch.sqrt(dims)
                        delta = (delta * magnitude.view(-1, 1, 1))
                    elif args.adv_norm_type == 'linf':
                        delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
                                                                     args.adv_init_mag) * input_mask.unsqueeze(2)
                else:
                    delta = torch.zeros_like(embedding_init)

                total_loss = 0.0
                for astep in range(args.adv_steps):
                    # (0) forward
                    delta.requires_grad_()
                    batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
                    # logits = model(**batch).logits
                    logits = model(**batch,return_dict=False)[0]
                    _, preds = logits.max(dim=-1)
                    # print(preds)
                    # print(logits)
                    # (1) backward
                    losses = F.cross_entropy(logits, labels.squeeze(-1))
                    loss = torch.mean(losses)
                    # todo Add the L-1 regularization loss to the loss fuction, weighted by
                    # `args.l1_loss_coef`. 这个应该在平均之后，还是之前？感觉是之前
                    # modified 会不会每个ministep都加，有点多了？但是后面也平均了
                    # 有trick，coef loss不应该被对抗，所以只在最后一步做，比把coef loss拿到外面反向传播，少做一次backward
                    loss = loss / args.adv_steps
                    if astep==args.adv_steps-1 and (args.l1_loss_coef > 0.0 or args.l1_loss_self_coef > 0.0):
                    # if astep==args.adv_steps-1 and (args.l1_loss_coef > 0.0 ):
                        l1_loss = 0.0
                        for m in model.modules():
                            if isinstance(m, BertSelfAttention) and m.self_slimming:
                                l1_loss += m.slimming_coef.abs().sum() * args.l1_loss_self_coef
                            if isinstance(m, BertLayer) and m.inter_slimming:
                                l1_loss += m.slimming_coef.abs().sum()* args.l1_loss_inter_coef
                        # logger.info("astep:{}, global step:{} l1_loss:{}, adv_loss_not_averaged:{}\n".format(astep,global_step,l1_loss ,loss*args.adv_steps))
                        loss += l1_loss
                                # * args.l1_loss_coef
                    total_loss += loss.item()
                    loss.backward()
                    # 拿出所有mask的grad
                    # all_coef_concat = None
                    # grad_list = [torch.reshape(m.slimming_coef.grad,(1,-1)) if (isinstance(m, BertSelfAttention) and m.self_slimming) or (isinstance(m, BertLayer) and m.inter_slimming) else None for m in model.modules()]
                    # for grad in grad_list:
                    #     if grad!=None:
                    #         if all_coef_concat==None:
                    #             all_coef_concat = grad
                    #         else:
                    #             all_coef_concat = torch.cat((all_coef_concat,grad),dim=1)
                    # writer.add_scalar(tag="Grad_norm/coef",scalar_value=torch.norm(all_coef_concat),global_step=global_step)

                    if astep == args.adv_steps - 1:
                        break

                    # (2) get gradient on delta
                    delta_grad = delta.grad.clone().detach()

                    # (3) update and clip。 denorm是用来归一化的，而delta_norm才是用来做范数约束的
                    if args.adv_norm_type == "l2":
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                        if args.adv_max_norm > 0:
                            delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                            exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                            reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                            delta = (delta * reweights).detach()
                    elif args.adv_norm_type == "linf":
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                                 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        delta = (delta + args.adv_lr * delta_grad / denorm).detach()

                    embedding_init = word_embedding_layer(input_ids)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                avg_loss.update(total_loss)
                pbar.set_description(f'epoch: {epoch: 0.4f}, '
                                     f'loss: {avg_loss.get_metric(): 0.4f}, '
                                     f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
                global_step+=1

                # modified
                # Record the learnable coefficients after each step of update
                # global self_slimming_coef_records, inter_slimming_coef_records
                idx_layer = 0
                for m in model.modules():
                    if isinstance(m, BertSelfAttention) and m.self_slimming:
                        self_slimming_coef_records[idx_layer].append(m.slimming_coef.detach().cpu().numpy().reshape(-1))
                        idx_layer += 1

                idx_layer = 0
                for m in model.modules():
                    if isinstance(m, BertLayer) and m.inter_slimming:
                        inter_slimming_coef_records[idx_layer].append(m.slimming_coef.detach().cpu().numpy().reshape(-1))
                        idx_layer += 1

            if num_training_steps < len(train_dataset)//args.bsz:

                s = Path(str(output_dir) + '/epoch' + str(epoch//1))
            else:
                s = Path(str(output_dir) + '/epoch' + str((epoch-1)//1))


            logger.info("!!!!!!!!!"+str(global_step)+" "+str(num_training_steps)+"  "+str(epoch)+"  "+str(epoch//1))
            if not s.exists():
                s.mkdir(parents=True)
            # model.save_pretrained(s)
            # tokenizer.save_pretrained(s)
            #
            # torch.save(args, os.path.join(s, "training_args.bin"))
            # logger.info("Saving model checkpoint to %s", output_dir)

            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            # logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.do_eval and not args.cal_time:
                logger.info('Evaluating...')
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for model_inputs, labels in dev_loader:
                        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                        labels = labels.to(device)
                        # logits = model(**model_inputs).logits
                        logits = model(**model_inputs,return_dict=False)[0]
                        _, preds = logits.max(dim=-1)
                        correct += (preds == labels.squeeze(-1)).sum().item()
                        total += labels.size(0)
                    accuracy = correct / (total + 1e-13)
                logger.info(f'Epoch: {epoch}, '
                            f'Loss: {avg_loss.get_metric(): 0.4f}, '
                            f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                            f'Accuracy: {accuracy}')

                if accuracy > best_accuracy:
                    logger.info('Best performance so far.')
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    best_accuracy = accuracy
                    best_dev_epoch = epoch
        logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')

    except KeyboardInterrupt:
        logger.info('Interrupted...')

    # Save the trained coefficients in ER.
    # Will be used to draw ER tickets later.
    if args.do_train:
        logger.info("Saving model coefficients to %s", output_dir)
        for i, self_slimming_coef in enumerate(self_slimming_coef_records):
            self_slimming_coef_records[i] = np.stack(self_slimming_coef, axis=0)
        np.save(os.path.join(args.output_dir, 'self_slimming_coef_records.npy'),
                np.stack(self_slimming_coef_records, axis=0))

        for i, inter_slimming_coef in enumerate(inter_slimming_coef_records):
            inter_slimming_coef_records[i] = np.stack(inter_slimming_coef, axis=0)
        np.save(os.path.join(args.output_dir, 'inter_slimming_coef_records.npy'),
                np.stack(inter_slimming_coef_records, axis=0))


    # test using best model
    if args.do_test:
        logger.info('Testing...')
        model = AutoModelForSequenceClassification.from_pretrained(output_dir, config=config)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for model_inputs, labels in test_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                # logits = model(**model_inputs).logits
                logits = model(**batch,return_dict=False)[0]
                _, preds = logits.max(dim=-1)
                correct += (preds == labels.squeeze(-1)).sum().item()
                total += labels.size(0)
            accuracy = correct / (total + 1e-13)
        logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
