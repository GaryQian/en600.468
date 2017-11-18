# -*- coding: utf-8 -*-
import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable

import numpy as np

from modelgary2 import NMT


_,_,_, src_vocab = torch.load(open('hw5.words', 'rb'))
_, _,_, trg_vocab = torch.load(open('hw5.phoneme', 'rb'))

#src_train, src_dev, src_test = src_train[:-1], src_dev[:-1], src_test[:-1]
#trg_train, trg_dev, trg_test = trg_train[1:], trg_dev[1:], trg_test[1:]


print trg_vocab.itos

src = []
with open("cmudict.words.tst" ,'r') as wrds:
  for line in wrds:
    src.append(torch.LongTensor([int(src_vocab.stoi[char]) for char in line.strip().split(" ") if char in src_vocab.stoi.keys()] ))
trg = []
with open("cmudict.phoneme.tst" ,'r') as wrds:
  for line in wrds:
    trg.append(torch.LongTensor([int(trg_vocab.stoi[char]) for char in line.strip().split(" ")] ))
    
epochId = 13
prob = '4.51'

nmt = torch.load(open('modeldump.nll_' + prob + '.epoch_' + str(epochId), 'rb'), pickle_module=dill, map_location=lambda storage, loc: storage)
results = []
for src, trg in zip(src,trg):
  results.append(nmt(Variable(src.unsqueeze(1)), Variable(trg.unsqueeze(1))).squeeze(0))

s = ""
for sent in results:
  s = ""
  for word in np.argmax(sent.data.numpy(),axis=0):
    s += trg_vocab.itos[word] + " "
  print s







"""
lines = []
with open('data/test.en.txt.cloze') as f:
    lines = f.read().splitlines()
    
    
_, _, _, vocab = torch.load(open("data/hw4_data.bin", 'rb'), pickle_module=dill)


vocabid = vocab.stoi

for line in lines:
  line = "<s> " + line + " </s>"
  vec = [[int(vocab.stoi[word]) for word in line.split(" ")]]
  blanks = []
  for i, w in enumerate(line.split(" ")):
    if w == "<blank>":
      blanks.append(i)
  #print vec
  result = rnn(Variable(torch.t(torch.Tensor(vec).long()))).data
  output = []
  #print result
  for i in blanks:
    output.append(result[i,0])
  #print blanks
  s = ""
  for dist in output:
    s += vocab.itos[np.argmax(dist.numpy())] + " "
  print s.encode('utf-8')
  
"""
