import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from model import BiRNNLM

epochId = 1
prob = '4.70'

rnn = torch.load(open('model.py.nll_' + prob + '.epoch_' + str(epochId), 'rb'), pickle_module=dill)

