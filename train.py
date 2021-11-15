# /usr/bin/env python
# coding=utf-8
"""train with valid"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
from transformers import BertConfig
import random
import logging
from tqdm import trange
import argparse

import utils
from optimization import BertAdam
from evaluate import evaluate
from dataloader import CustomDataLoader
from model import BertForRE

import json
import pandas as pd

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2021, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default="deafult")
parser.add_argument('--corpus_type', type=str, default="WebNLG", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--epoch_num', type=int, default=10, help="number of epochs")
parser.add_argument('--multi_gpu', action='store_true', help="ensure multi-gpu training")
parser.add_argument('--restore_file', default=None, help="name of the file containing weights to reload")

parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")

parser.add_argument('--ensure_cross', type=str, default='none', help="['none'(default),, 'avg']")
parser.add_argument('--ensure_rel', type=str, default='default', help="['default', 'hgat', '_astoken', '_relemb', '_concat']")
parser.add_argument('--ensure_corres', type=str, default='default', help="correspondence ablation")
parser.add_argument('--use_symmetries', type=str, default='none', help="symmetries relations, ['none'(default), 'symmetries', 'symmetries_rate']")
parser.add_argument('--sent_rels', type=str, default='none', help="sentense relations, ['none'(default), 'global', 'contextual', 'contextual_global']")
parser.add_argument('--sent_attn', type=str, default='none', help="sentense attention, ['none'(default)]")
parser.add_argument('--cross_data', type=int, default=0, help="Extract implied relation")

parser.add_argument('--use_feature_enchanced', action='store_true', help="use MLP to enchancing feature representation of subject and object")


parser.add_argument('--num_negs', type=int, default=4,
                    help="number of negative sample when ablate relation judgement")


def train(model, data_iterator, optimizer, params, ex_params):
    """Train the model one epoch
    """
    # set model to training mode
    model.train()

    loss_avg = utils.RunningAverage()
    loss_avg_seq = utils.RunningAverage()
    loss_avg_mat = utils.RunningAverage()
    loss_avg_rel = utils.RunningAverage()

    # Use tqdm for progress bar
    # one epoch
    t = trange(len(data_iterator), ascii=True)
    for step, _ in enumerate(t):
        # fetch the next training batch
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batch

        # compute model output and loss
        loss, loss_seq, loss_mat, loss_rel = model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                                   potential_rels=relations, corres_tags=corres_tags, rel_tags=rel_tags,
                                                   ex_params=ex_params)

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps

        # back-prop
        loss.backward()

        if (step + 1) % params.gradient_accumulation_steps == 0:
            # performs updates using calculated gradients
            optimizer.step()
            model.zero_grad()

        # update the average loss
        loss_avg.update(loss.item() * params.gradient_accumulation_steps)
        loss_avg_seq.update(loss_seq.item())
        loss_avg_mat.update(loss_mat.item())
        loss_avg_rel.update(loss_rel.item())
        # 右边第一个0为填充数，第二个5为数字个数为5位，第三个3为小数点有效数为3，最后一个f为数据类型为float类型。
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()),
                      loss_seq='{:05.3f}'.format(loss_avg_seq()),
                      loss_mat='{:05.3f}'.format(loss_avg_mat()),
                      loss_rel='{:05.3f}'.format(loss_avg_rel()))
    logging.info("loss={:05.3f}, loss_seq={:05.3f}, loss_mat={:05.3f}, loss_rel={:05.3f}".format(loss_avg(), loss_avg_seq(), loss_avg_mat(), loss_avg_rel()))


def train_and_evaluate(model, params, ex_params, restore_file=None):
    """Train the model and evaluate every epoch."""
    # Load training data and val data
    dataloader = CustomDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train', ex_params=ex_params)
    val_loader = dataloader.get_dataloader(data_sign='val', ex_params=ex_params)
    test_loader = dataloader.get_dataloader(data_sign='test', ex_params=ex_params)

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        # 读取checkpoint
        model, optimizer = utils.load_checkpoint(restore_path)

    model.to(params.device)
    # parallel model
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    # fine-tuning
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [
        # pretrain model param
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.fin_tuning_lr
         },
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.fin_tuning_lr
         },
        # downstream model
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.downs_en_lr
         },
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.downs_en_lr
         }
    ]
    num_train_optimization_steps = len(train_loader) // params.gradient_accumulation_steps * args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=params.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=params.clip_grad)

    # patience stage
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))

        # Train for one epoch on training set
        train(model, train_loader, optimizer, params, ex_params)

        # Evaluate for one epoch on training set and validation set
        # train_metrics = evaluate(args, model, train_loader, params, mark='Train',
        #                          verbose=True)  # Dict['loss', 'f1']
        val_metrics, _, _ = evaluate(model, val_loader, params, ex_params, mark='Val')
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'model': model_to_save,
                               'optim': optimizer_to_save},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(params.ex_dir / 'params.json')

        # stop training based params.patience
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            logging.warning("Best val f1: {}".format(best_val_f1))
            break


    logging.info("\n\nTest in TestSet:\n")
    metrics, predictions, ground_truths, pred_rel, error, lack = evaluate(model, test_loader, params, ex_params, mark='test')

    idx2rel = dict(zip(params.rel2idx.values(), params.rel2idx.keys()))

    for i in range(len(pred_rel)):
        pred_rel[i] = list(map(lambda x: idx2rel[x], pred_rel[i]))
    for i in range(len(ground_truths)):
        predictions[i] = list(map(lambda x: (x[0], x[1], idx2rel[x[2]]), predictions[i]))
        ground_truths[i] = list(map(lambda x: (x[0], x[1], idx2rel[x[2]]), ground_truths[i]))
    for i in range(len(error)):
        error[i] = list(map(lambda x: (x[0], x[1], idx2rel[x[2]]), error[i]))
    for i in range(len(lack)):
        lack[i] = list(map(lambda x: (x[0], x[1], idx2rel[x[2]]), lack[i]))

    rel_lack = []       # 关系缺失
    rel_error = []      # 关系错误
    rel_error_triples = []
    for p, g, r, e, l in zip(predictions, ground_truths, pred_rel, error, lack):
        rel_p = set(r)
        rel_g = set([i[2] for i in g])
        rel_lack.append(list(rel_g - rel_p))
        rel_error.append(list(rel_p - rel_g))
        rel_error_triple = []
        for e1, e2, r1 in e:
            for ge1, ge2, gr1 in g:
                if e1==ge1 and e2 == ge2:
                    rel_error_triple.append((e1, e2, "{}/{}".format(r1, gr1)))
                    break
        rel_error_triples.append(rel_error_triple)

    with open(params.data_dir / f'test_triples.json', 'r', encoding='utf-8') as f_src:
        src = json.load(f_src)
        df = pd.DataFrame(
            {
                'text': [sample['text'] for sample in src],
                'pre': predictions,
                'truth': ground_truths,
                'rel': pred_rel,
                'label': [set(p) == set(g) for p, g in zip(predictions, ground_truths)],
                'error': error,
                'rel_error': rel_error,
                'rel_error_triples': rel_error_triples,
                'lack': lack,
                'rel_lack': rel_lack,
            }
        )
        df = df.mask(df.applymap(str).eq('[]'))
        df.to_csv(os.path.join(params.ex_dir,'{:.4f}_test_result.csv'.format(metrics['f1'])))
    logging.info('-done')



def test_with_restore_file(params, ex_params, restore_file='best'):
    
    # Load training data and val data
    dataloader = CustomDataLoader(params)
    test_loader = dataloader.get_dataloader(data_sign='test', ex_params=ex_params)

    restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
    logging.info("Restoring parameters from {}".format(restore_path))
    # 读取checkpoint
    model, optimizer = utils.load_checkpoint(restore_path)

    model.to(params.device)
    # parallel model
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    logging.info("\n\nTest in TestSet {}:\n".format(restore_file))
    metrics, predictions, ground_truths, pred_rel, error, lack = evaluate(model, test_loader, params, ex_params, mark='test')

    idx2rel = dict(zip(params.rel2idx.values(), params.rel2idx.keys()))

    for i in range(len(pred_rel)):
        pred_rel[i] = list(map(lambda x: idx2rel[x], pred_rel[i]))
    for i in range(len(ground_truths)):
        predictions[i] = list(map(lambda x: (x[0], x[1], idx2rel[x[2]]), predictions[i]))
        ground_truths[i] = list(map(lambda x: (x[0], x[1], idx2rel[x[2]]), ground_truths[i]))
    for i in range(len(error)):
        error[i] = list(map(lambda x: (x[0], x[1], idx2rel[x[2]]), error[i]))
    for i in range(len(lack)):
        lack[i] = list(map(lambda x: (x[0], x[1], idx2rel[x[2]]), lack[i]))

    rel_lack = []       # 关系缺失
    rel_error = []      # 关系错误
    rel_error_triples = []
    for p, g, r, e, l in zip(predictions, ground_truths, pred_rel, error, lack):
        rel_p = set(r)
        rel_g = set([i[2] for i in g])
        rel_lack.append(list(rel_g - rel_p))
        rel_error.append(list(rel_p - rel_g))
        rel_error_triple = []
        for e1, e2, r1 in e:
            for ge1, ge2, gr1 in g:
                if e1==ge1 and e2 == ge2:
                    rel_error_triple.append((e1, e2, "{}/{}".format(r1, gr1)))
                    break
        rel_error_triples.append(rel_error_triple)

    with open(params.data_dir / f'test_triples.json', 'r', encoding='utf-8') as f_src:
        src = json.load(f_src)
        df = pd.DataFrame(
            {
                'text': [sample['text'] for sample in src],
                'pre': predictions,
                'truth': ground_truths,
                'rel': pred_rel,
                'label': [set(p) == set(g) for p, g in zip(predictions, ground_truths)],
                'error': error,
                'rel_error': rel_error,
                'rel_error_triples': rel_error_triples,
                'lack': lack,
                'rel_lack': rel_lack,
            }
        )
        df = df.mask(df.applymap(str).eq('[]'))
        df.to_csv(os.path.join(params.ex_dir,'{}_{:.4f}_result.csv'.format(restore_file, metrics['f1'])))
    logging.info('-done')



if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(args.ex_index, args.corpus_type)
    utils.set_logger(save=True, log_path=os.path.join(params.ex_dir, 'train.log'))
    ex_params = vars(args)
    utils.print_config(ex_params)
    logging.info(params.get_params_info())

    if args.multi_gpu:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    else:
        torch.cuda.set_device(args.device_id)
        print('current device:', torch.cuda.current_device())
        params.n_gpu = n_gpu = 1

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Set the logger
    logging.info(f"Model type:")
    logging.info("device: {}".format(params.device))

    logging.info('Load pre-train model weights...')
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'config.json'))
    model = BertForRE(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params, ex_params=ex_params)
    logging.info('-done')

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(model, params, ex_params, args.restore_file)
    
    test_with_restore_file(params, ex_params, 'best')
    test_with_restore_file(params, ex_params, 'last')
