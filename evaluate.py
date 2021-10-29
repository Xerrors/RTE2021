# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import json
import logging
from pickle import NONE
import random
import argparse

from tqdm import tqdm
import os

import torch
import numpy as np
import pandas as pd

from metrics import tag_mapping_nearest, tag_mapping_corres
from utils import Label2IdxSub, Label2IdxObj
import utils
from dataloader import CustomDataLoader

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2021, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--mode', type=str, default="test") # zwj add
parser.add_argument('--corpus_type', type=str, default="NYT", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--restore_file', default='last', help="name of the file containing weights to reload")

parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")

parser.add_argument('--ensure_cross', type=str, default='none', help="['none', 'avg']")
parser.add_argument('--ensure_rel', type=str, default='default', help="['default', 'hgat', '_astoken', '_relemb']")
parser.add_argument('--ensure_corres', type=str, default='default', help="correspondence ablation")

parser.add_argument('--num_negs', type=int, default=4,
                    help="number of negative sample when ablate relation judgement")


def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }


def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output


def evaluate(model, data_iterator, params, ex_params, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    rel_num = params.rel_num

    predictions = []
    ground_truths = []
    # >>> zwj add start
    pred_rel = []
    error = []
    lack = []
    # <<< zwj add end
    correct_num, predict_num, gold_num = 0, 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, triples, input_tokens = batch
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)

            # (sum(x_i), seq_len)
            pred_seqs = pred_seqs.detach().cpu().numpy()
            # (bs, seq_len, seq_len)
            pre_corres = pre_corres.detach().cpu().numpy()
        if ex_params['ensure_rel']:
            # (bs,)
            xi = np.array(xi)
            # (sum(s_i),)
            pred_rels = pred_rels.detach().cpu().numpy()
            # decode by per batch
            xi_index = np.cumsum(xi).tolist()
            # (bs+1,)
            xi_index.insert(0, 0)

        for idx in range(bs):
            if ex_params['ensure_rel']:
                pred_rel.append(pred_rels[xi_index[idx]:xi_index[idx + 1]])
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                 pre_corres=pre_corres[idx],
                                                 pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            else:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[idx * rel_num:(idx + 1) * rel_num],
                                                 pre_corres=pre_corres[idx],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)

            gold_triples = span2str(triples[idx], input_tokens[idx])
            pre_triples = span2str(pre_triples, input_tokens[idx])
            gold_triples = set(gold_triples)
            pre_triples = set(pre_triples)
            sort_key = lambda x: " ".join(list(map(str, x)))
            ground_truths.append(sorted(list(gold_triples),key=sort_key))
            predictions.append(sorted(list(pre_triples), key=sort_key))
            error.append(sorted(list(pre_triples - gold_triples),key=sort_key))
            lack.append(sorted(list(gold_triples - pre_triples), key=sort_key))
            # counter
            correct_num += len(pre_triples & gold_triples)
            predict_num += len(pre_triples)
            gold_num += len(gold_triples)
    metrics = get_metrics(correct_num, predict_num, gold_num)
    # logging loss, f1 and report
    metrics_str = "; ".join("{}: {:05.4f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics:\n".format(mark) + metrics_str)
    if mark == "Val":
        return metrics, predictions, ground_truths
    else:
        return metrics, predictions, ground_truths, pred_rel, error, lack


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index, corpus_type=args.corpus_type)
    logger_path = os.path.join(params.ex_dir, 'train.log')
    ex_params = vars(args)

    torch.cuda.set_device(args.device_id)
    print('current device:', torch.cuda.current_device())
    mode = args.mode
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, args.restore_file)}.pth.tar')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    logging.info('-done')

    logging.info("Starting prediction...")
    metrics, predictions, ground_truths, pred_rel, error, lack = evaluate(model, loader, params, ex_params, mark=mode)

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

    with open(logger_path, 'a') as f:
        f.write("\n\nTest in TestSet:\n")
        f.write("; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items()))

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

    with open(params.data_dir / f'{mode}_triples.json', 'r', encoding='utf-8') as f_src:
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
        df.to_csv(params.ex_dir / f'{mode}_result.csv')
    logging.info('-done')