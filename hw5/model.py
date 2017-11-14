import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import math

# TODO: Your implementation goes here
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    
    self.initParams = torch.load(open("model.param", "rb"))
    
    
    self.embedding = torch.nn.Embedding(36616, 300)
    self.embedding.weight.data = self.initParams["encoder.embeddings.emb_luts.0.weight"]
    
    self.lstm = torch.nn.LSTM(input_size=2048, hidden_size=300, num_layers=1, bidirectional=True)
    
    self.lstm.weight_ih_l0.data = self.initParams["encoder.rnn.weight_ih_l0"]
    self.lstm.weight_hh_l0.data = self.initParams["encoder.rnn.weight_hh_l0"]
    self.lstm.bias_ih_l0.data = self.initParams["encoder.rnn.bias_ih_l0"]
    self.lstm.bias_hh_l0.data = self.initParams["encoder.rnn.bias_hh_l0"]
    
    self.lstm.weight_ih_l0_reverse.data = self.initParams["encoder.rnn.weight_ih_l0_reverse"]
    self.lstm.weight_hh_l0_reverse.data = self.initParams["encoder.rnn.weight_hh_l0_reverse"]
    self.lstm.bias_ih_l0_reverse.data = self.initParams["encoder.rnn.bias_ih_l0_reverse"]
    self.lstm.bias_hh_l0_reverse.data = self.initParams["encoder.rnn.bias_hh_l0_reverse"]
    

  def forward(self, input, do_dropout=False):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    output, hidden = self.lstm(output, None)
    return output
    
m = NMT()