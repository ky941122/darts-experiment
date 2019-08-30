#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date   : 2019-08-26
# @Author : KangYu
# @File   : voice_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sequence_mask(lengths, maxlen):
    batch_size = len(lengths)
    lengths = torch.tensor(lengths).unsqueeze(-1).cuda().float()
    maxlen = torch.arange(maxlen).repeat(batch_size, 1).cuda().float()
    mask = maxlen < lengths
    mask = mask.float()
    return mask


def mask(inputs, sent_nums, max_sent_nums, mode):
    mask = sequence_mask(sent_nums, max_sent_nums)
    for _ in range(len(inputs.shape) - 2):
        mask = torch.unsqueeze(mask, 2).float()
    if mode == 'mul':
        return inputs * mask
    elif mode == 'add':
        return inputs - (1 - mask) * 1e12
    else:
        raise ValueError("mode is not in [mul, add]!")


class Voice_Attention(nn.Module):

    def __init__(self, vocab_size, config):
        super().__init__()

        w2v_np = np.load(config.pretrained_w2v_path)
        w2v_np = np.concatenate([np.array([[0.0] * config.embedding_dim]), w2v_np], axis=0)
        w2v = torch.tensor(w2v_np)

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0).from_pretrained(w2v)
        self.dense1 = nn.Linear(config.voice_embed_dim, 32)
        self.dense2 = nn.Linear(config.embedding_dim*2, config.embedding_dim)
        self.config = config

    def forward(self, word_idx, voice_embeded, sent_nums, max_sent_nums):

        word_embeded = self.embedding(word_idx)   #[batch, sent_nums, word_nums, embed_dims]
        voice_Q = self.dense1(voice_embeded)
        voice_K = self.dense1(voice_embeded)
        voice_K = voice_K.transpose(1, 2)
        voice_A = torch.matmul(voice_Q, voice_K)
        voice_A = voice_A.transpose(1, 2)
        voice_A = mask(voice_A, sent_nums, max_sent_nums, mode='add')
        voice_A = voice_A.transpose(1, 2)
        voice_A = F.softmax(voice_A, dim=-1)

        word_V = word_embeded.reshape((self.config.batch_size, max_sent_nums, -1))  #[batch, sent_nums, word_nums * embed_dims]
        voice_O = torch.matmul(voice_A, word_V) #[batch, sent_nums, word_nums * embed_dims]
        voice_O = mask(voice_O, sent_nums, max_sent_nums, mode='mul')
        voice_O = voice_O.reshape((self.config.batch_size, max_sent_nums, self.config.max_word_nums, self.config.embedding_dim)) #[batch, sent_nums, word_nums, embed_dims]
        O = torch.cat((voice_O, word_embeded), dim=-1) #[batch, sent_nums, word_nums, 2*embed_dims]
        O = self.dense2(O)

        O = O.transpose(1, 3).transpose(2, 3)  #[batch, embed_dims, sent_nums, word_nums] as [batch, C, H, W]

        return O, voice_A


