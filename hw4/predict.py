# -*- coding: utf-8 -*-
import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from model import BiRNNLM

import numpy as np

epochId = 2
prob = '3.94'

rnn = torch.load(open('model.py.nll_' + prob + '.epoch_' + str(epochId), 'rb'), pickle_module=dill)

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
  
  