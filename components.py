from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

class HGAT(nn.Module):

    def __init__(self, hidden_size, class_num):
        super(HGAT, self).__init__()
        self.class_num = class_num
        self.embeding = nn.Embedding(class_num, hidden_size)
        self.relation = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList([GATLayer(hidden_size) for _ in range(2)])

    def forward(self, x, mask=None):
        p = torch.arange(self.class_num).long()
        if torch.cuda.is_available():
            p = p.cuda()
        p = self.relation(self.embeding(p))
        p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))  # bsz classnum dim 
        x, p = self.gat_layer(x, p, mask)  # x bcd
        return x, p
    
    def get_relation_emb(self, p):
        return self.relation(self.embeding(p))

    def gat_layer(self, x, p, mask=None):

        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class GATLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.ra1 = RelationAttention(hidden_size)
        self.ra2 = RelationAttention(hidden_size)

    def forward(self, x, p, mask=None):
        x_ = self.ra1(x, p)
        x = x_ + x
        p_ = self.ra2(p, x, mask)
        p = p_ + p
        return x, p


class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        q = self.query(p)
        k = self.key(x)
        score = self.fuse(q, k)
        if mask is not None:
            mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
            score = score.masked_fill(mask == 1, -1e9)
        score = F.softmax(score, 2)
        v = self.value(x)
        out = torch.einsum('bcl,bld->bcd', score, v) + p
        g = self.gate(torch.cat([out, p], 2)).sigmoid()
        out = g * out + (1 - g) * p
        return out

    def fuse(self, x, y):
        x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        temp = torch.cat([x, y], 3)
        # TODO：这里是不是应该加一个激活函数 ReLU 之类的
        return self.score(temp).squeeze(3)


def cross_conv(x):
    l = x.size(1)
    a = x.mean(1).unsqueeze(1).repeat(1,l,1,1)
    b = x.mean(2).unsqueeze(2).repeat(1,1,l,1)
    return a + b


class MultiheadAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.batch_first = batch_first

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if self.batch_first:
            query = query.transpose(0,1)
            key = key.transpose(0,1)
            value = value.transpose(0,1)

        output, _  = self.attn(query, key, value, key_padding_mask, need_weights, attn_mask)

        if self.batch_first:
            output = output.transpose(0,1)
        return output
