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
    

  def forward(self, input, hidden=None):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    output, hidden = self.lstm(output, hidden)
    return output, hidden


class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.initParams = torch.load(open("model.param", "rb"))
    
    self.embedding = torch.nn.Embedding(23262, 300)
    self.embedding.weight.data = self.initParams["decoder.embeddings.emb_luts.0.weight"]
    
    self.lstm = torch.nn.LSTM(input_size=4096, hidden_size=1024, num_layers=1)
    
    self.lstm.weight_ih_l0.data = self.initParams["decoder.rnn.layers.0.weight_ih"]
    self.lstm.weight_hh_l0.data = self.initParams["decoder.rnn.layers.0.weight_hh"]
    self.lstm.bias_ih_l0.data = self.initParams["decoder.rnn.layers.0.bias_ih"]
    self.lstm.bias_hh_l0.data = self.initParams["decoder.rnn.layers.0.bias_hh"]
    
    self.attin = torch.nn.Linear(1024, 1024)
    self.attin.weight.data = self.initParams["decoder.attn.linear_in.weight"]
    
    self.softmax = nn.LogSoftmax()
    
    self.attout = torch.nn.Linear(1024, 2048)
    self.attout.weight.data = self.initParams["decoder.attn.linear_out.weight"]
    

  def forward(self, input, hidden=None):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    output, hidden = self.lstm(output, hidden)
    output = self.softmax(self.attin(output))
    output = self.attout(output)
    return output, hidden
    

m = Encoder()
m = Decoder()