import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import math

# TODO: Your implementation goes here
class Encoder(nn.Module):
  def __init__(self, vocab_size):
    super(Encoder, self).__init__()

    self.embedding = torch.nn.Embedding(vocab_size, 300)

    self.lstm = torch.nn.LSTM(input_size=300, hidden_size=512, num_layers=1, bidirectional=True)
    
    self.hidden = None
    

  def forward(self, input):
    embedded = self.embedding(input)
    output = embedded
    output, self.hidden = self.lstm(output, self.hidden)
    return output, self.hidden


class Decoder(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size):
    super(Decoder, self).__init__()
    
    self.initParams = torch.load(open("model.param", "rb"))
    
    self.embedding = torch.nn.Embedding(trg_vocab_size, 300)
    
    self.attin = torch.nn.Linear(1024, 1024)
    self.softmax = nn.LogSoftmax()
    
    self.attout = torch.nn.Linear(2048, 1024)
    
    
    self.lstm = torch.nn.LSTM(input_size=1324, hidden_size=1024, num_layers=1)
    
    #23262
    self.gen = torch.nn.Linear(1024,trg_vocab_size)
    self.hidden = Parameter(torch.randn((48, 1024)))
    

  def forward(self, targ, encoder_out):
    embedded = self.embedding(targ)
    output = embedded
    
    sc = Variable(torch.zeros((len(encoder_out),48,)))
    for i in range(len(encoder_out)):
      sc[i] = self.score(encoder_out[i], self.hidden)
    a = self.softmax(sc)
    a = a.repeat(1024,1,1).transpose(0,1).transpose(1,2)

    mult = torch.mul(a, encoder_out)

    s = torch.sum(mult, 0)

    context = torch.tanh(self.attout(torch.cat((s, self.hidden), 1)))

    output = torch.cat((context.repeat(len(embedded), 1,1), embedded), 2)
    output, hiddenN = self.lstm(output, Variable(torch.zeros((2, 48, 1024,))))
    generated = self.gen(output)
    return generated
  
  def score(self, h_s, h_t):
    h_t = self.attin(h_t)
    a = Variable(torch.zeros((48,1)))
    for i in range(len(h_t)):
        a[i] = torch.dot(h_s[i], h_t[i])
    return a.transpose(0,1)

    
    
class NMT(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size):
    super(NMT, self).__init__()
    self.Encoder = Encoder(src_vocab_size)
    self.Decoder = Decoder(src_vocab_size, trg_vocab_size)
    self.enchidden = None

  def forward(self, input, targ):
    
    encout, _ = self.Encoder(input)
    
    return self.Decoder(targ, encout)