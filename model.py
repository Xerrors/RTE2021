# /usr/bin/env python
# coding=utf-8
"""model"""
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from transformers import BertModel
from components import HGAT, cross_conv, MultiheadAttention

import logging


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output

class MultiNonLinearClassifierWithCrossAvg(nn.Module):
    """ TODO: 重新考虑可行性
    
    """
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifierWithCrossAvg, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = cross_conv(features_tmp)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class SequenceLabelForSO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output



class BertForRE(nn.Module):
    def __init__(self, config, pretrained_model_name_or_path, params, ex_params):
        super().__init__()
        self.max_seq_len = params.max_seq_length
        self.seq_tag_size = params.seq_tag_size
        self.rel_num = params.rel_num

        if 'hgat' in ex_params['ensure_rel']:
            self.gat = HGAT(config.hidden_size, params.rel_num)
            self.gat_attention = nn.MultiheadAttention(config.hidden_size, 8)
            self.get_relation_linear = nn.Linear(config.hidden_size, 1)

        # pretrain model
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path) # zwj add
        # self.bert = BertModel(config)

        # sequence tagging
        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.sequence_tagging_sum = SequenceLabelForSO(config.hidden_size, self.seq_tag_size, params.drop_prob)
        # global correspondence
        if ex_params['ensure_cross'] == 'avg':
            self.global_corres_cross_avg = MultiNonLinearClassifierWithCrossAvg(config.hidden_size * 2, 1, params.drop_prob)

        if ex_params['sent_attn'] != "none":
            self.sent_attention = MultiheadAttention(config.hidden_size, 8, dropout=params.drop_prob, batch_first=True)

        if ex_params['sent_rels'] == 'global':
            self.sent_global_attention = MultiheadAttention(config.hidden_size, 8, dropout=params.drop_prob, batch_first=True)
            self.drop_dim_21 = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.FFNN = nn.Linear(config.hidden_size, config.hidden_size)

        if ex_params['sent_rels'] == 'global2':
            self.sent_global_attention = MultiheadAttention(config.hidden_size, 8, dropout=params.drop_prob, batch_first=True)
            self.drop_dim_21 = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.FFNN = nn.Linear(config.hidden_size, config.hidden_size)
            self.FFNN2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout1 = nn.Dropout(params.drop_prob)
            self.dropout2 = nn.Dropout(params.drop_prob)
            self.dropout3 = nn.Dropout(params.drop_prob)
            self.norm1 = nn.LayerNorm(config.hidden_size)
            self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # https://github.com/neukg/GRTE/blob/main/model.py#L62 2021.11.15
        if ex_params['use_feature_enchanced']:
            self.Lr_e1=nn.Linear(config.hidden_size,config.hidden_size)
            self.Lr_e2=nn.Linear(config.hidden_size,config.hidden_size)

        self.global_corres = MultiNonLinearClassifier(config.hidden_size * 2, 1, params.drop_prob)
        # relation judgement
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size, params.rel_num, params.drop_prob)
        self.rel_embedding = nn.Embedding(params.rel_num, config.hidden_size)

        # self.init_weights()

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            seq_tags=None,
            potential_rels=None,
            corres_tags=None,
            rel_tags=None,
            ex_params=None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            rel_tags: (bs, rel_num)
            potential_rels: (bs,), only in train stage.
            seq_tags: (bs, 2, seq_len)
            corres_tags: (bs, seq_len, seq_len)
            ex_params: experiment parameters
        """
        # get params for experiments
        corres_threshold, rel_threshold = ex_params.get('corres_threshold', 0.5), ex_params.get('rel_threshold', 0.1)
        # ablation study
        ensure_corres, ensure_rel = ex_params['ensure_corres'], ex_params['ensure_rel']
        ensure_cross = ex_params['ensure_cross'] # zwj add
        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            # output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        bs, seq_len, h = sequence_output.size()

        is_inference = seq_tags is None

        if ensure_rel == 'default':
            # (bs, h)
            h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
            # (bs, rel_num)
            rel_pred = self.rel_judgement(h_k_avg)

        elif 'hgat' in ensure_rel:
            # (bs, h)
            gat_hidden, _ = self.gat(sequence_output, attention_mask) # h(bsz, n, dim), r(bsz, m, dim)
            h_k_avg = self.masked_avgpool(gat_hidden, attention_mask)
            # (bs, rel_num)
            rel_pred = self.rel_judgement(h_k_avg)
            if 'astoken' in ensure_rel:
                sequence_output = gat_hidden

        # before fuse relation representation
        if ensure_corres == 'default':
            # for every position $i$ in sequence, should concate $j$ to predict.
            if ex_params['use_feature_enchanced']:
                sub_extend = nn.ELU()(self.Lr_e1(sequence_output)).unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
                obj_extend = nn.ELU()(self.Lr_e2(sequence_output)).unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)
            else:
                sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
                obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)

            # if ex_params['sent_attn'] == 'span':
            #     self.sent_attention(sub_extend.unsqueeze(0))
            # batch x seq_len x seq_len x 2*hidden
            corres_pred = torch.cat([sub_extend, obj_extend], 3)

            if ex_params['sent_rels'] == 'contextual_global':
                logging.warning("Empty for this option '{}', and we will continue running with default setting('{}')".format(ex_params['sent_rels'], "none"))
                corres_pred = corres_pred # pass

            elif ex_params['sent_rels'] == 'global':
                corres_pred = nn.ReLU()(self.drop_dim_21(corres_pred))
                seq_attn = self.sent_global_attention(corres_pred.view(bs, seq_len*seq_len, h), sequence_output, sequence_output)
                seq_attn = nn.ReLU()(self.FFNN(seq_attn))
                corres_pred = torch.cat([corres_pred, seq_attn.view(bs, seq_len, seq_len, h)], 3)

            elif ex_params['sent_rels'] == 'global2':
                corres_pred = nn.ReLU()(self.drop_dim_21(corres_pred))
                x = corres_pred.view(bs, seq_len*seq_len, h)

                attn = self.dropout3(
                    self.sent_global_attention(
                        x,
                        sequence_output,
                        sequence_output))
                x = self.norm1(x + attn)
                ff = self.dropout2(self.FFNN2(self.dropout1(nn.ReLU()(self.FFNN(x)))))
                x = self.norm2(x + ff)

                corres_pred = torch.cat([corres_pred, x.view(bs, seq_len, seq_len, h)], 3)

            elif ex_params['sent_rels'] == 'contextual':
                logging.warning("Empty for this option '{}', and we will continue running with default setting('{}')".format(ex_params['sent_rels'], "none"))
                corres_pred = corres_pred # pass

            # (bs, seq_len, seq_len)
            if ensure_cross == 'avg':
                corres_pred = self.global_corres_cross_avg(corres_pred).squeeze(-1)
            else:
                corres_pred = self.global_corres(corres_pred).squeeze(-1)

            mask_tmp1 = attention_mask.unsqueeze(-1)
            mask_tmp2 = attention_mask.unsqueeze(1)
            corres_mask = mask_tmp1 * mask_tmp2

        # relation predict and data construction in inference stage
        xi, pred_rels = None, None
        if ensure_rel and is_inference:
            # (bs, rel_num)
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > rel_threshold,
                                          torch.ones(rel_pred.size(), device=rel_pred.device),
                                          torch.zeros(rel_pred.size(), device=rel_pred.device))

            # if potential relation is null
            for idx, sample in enumerate(rel_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)
                    max_index = torch.argmax(rel_pred[idx])
                    sample[max_index] = 1
                    rel_pred_onehot[idx] = sample

            # 2*(sum(x_i),)
            bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot, as_tuple=True)
            # get x_i
            xi_dict = Counter(bs_idxs.tolist())
            xi = [xi_dict[idx] for idx in range(bs)]

            pos_seq_output = []
            pos_potential_rel = []
            pos_attention_mask = []
            for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                # (seq_len, h)
                pos_seq_output.append(sequence_output[bs_idx])
                pos_attention_mask.append(attention_mask[bs_idx])
                pos_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            sequence_output = torch.stack(pos_seq_output, dim=0)
            # (sum(x_i), seq_len)
            attention_mask = torch.stack(pos_attention_mask, dim=0)
            # (sum(x_i),)
            potential_rels = torch.stack(pos_potential_rel, dim=0)
        # ablation of relation judgement
        elif not ensure_rel and is_inference:
            # construct test data
            sequence_output = sequence_output.repeat((1, self.rel_num, 1)).view(bs * self.rel_num, seq_len, h)
            attention_mask = attention_mask.repeat((1, self.rel_num)).view(bs * self.rel_num, seq_len)
            potential_rels = torch.arange(0, self.rel_num, device=input_ids.device).repeat(bs)

        # (bs/sum(x_i), h)
        if 'relemb' in ensure_rel:
            rel_emb = self.gat.get_relation_emb(potential_rels)
        else:
            rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)
        if ex_params['emb_fusion'] == 'concat':
            # (bs/sum(x_i), seq_len, 2*h)
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub = self.sequence_tagging_sub(decode_input)
            output_obj = self.sequence_tagging_obj(decode_input)
        elif ex_params['emb_fusion'] == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub, output_obj = self.sequence_tagging_sum(decode_input)

        # train
        if not is_inference:
            # calculate loss
            attention_mask = attention_mask.view(-1)
            # sequence label loss
            loss_func = nn.CrossEntropyLoss(reduction='none')
            loss_seq_sub = (loss_func(output_sub.view(-1, self.seq_tag_size),
                                      seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq_obj = (loss_func(output_obj.view(-1, self.seq_tag_size),
                                      seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq = (loss_seq_sub + loss_seq_obj) / 2
            # init
            loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
            if ensure_corres:
                corres_pred = corres_pred.view(bs, -1)
                corres_mask = corres_mask.view(bs, -1)
                corres_tags = corres_tags.view(bs, -1)
                loss_func = nn.BCEWithLogitsLoss(reduction='none')
                loss_matrix = (loss_func(corres_pred,
                                         corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

            if ensure_rel:
                loss_func = nn.BCEWithLogitsLoss(reduction='mean')
                loss_rel = loss_func(rel_pred, rel_tags.float())

            loss = loss_seq + loss_matrix + loss_rel
            return loss, loss_seq, loss_matrix, loss_rel
        # inference
        else:
            # (sum(x_i), seq_len)
            pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
            pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
            # (sum(x_i), 2, seq_len)
            pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
            if ensure_corres:
                corres_pred = torch.sigmoid(corres_pred) * corres_mask
                # (bs, seq_len, seq_len)
                pred_corres_onehot = torch.where(corres_pred > corres_threshold,
                                                 torch.ones(corres_pred.size(), device=corres_pred.device),
                                                 torch.zeros(corres_pred.size(), device=corres_pred.device))
                return pred_seqs, pred_corres_onehot, xi, pred_rels
            return pred_seqs, xi, pred_rels


if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'config.json'))
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    model.to(params.device)

    for n, _ in model.named_parameters():
        print(n)
