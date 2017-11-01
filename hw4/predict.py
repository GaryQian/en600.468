import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from model import BiRNNLM

epochId = 2
prob = '4.60'

rnn = torch.load(open('model.py.nll_' + prob + '.epoch_' + str(epochId), 'rb'), pickle_module=dill)

lines = []
with open('data/test.en.txt.cloze') as f:
    lines = f.read().splitlines()
    
    
_, _, _, vocab = torch.load(open("data/hw4_data.bin", 'rb'), pickle_module=dill)


vocabid = vocab.stoi

for line in lines:
  vec = [[vocab.stoi(word) for word in line.split(" ")]]
  output = rnn(vec)