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

    self.embedding = torch.nn.Embedding(36616, 300)

    self.lstm = torch.nn.LSTM(input_size=300, hidden_size=512, num_layers=1, bidirectional=True)
    
    self.hidden = None
    

  def forward(self, input):
    embedded = self.embedding(input)
    output = embedded
    output, self.hidden = self.lstm(output, self.hidden)
    return output, self.hidden


class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.initParams = torch.load(open("model.param", "rb"))
    
    self.embedding = torch.nn.Embedding(23262, 300)
    
    self.attin = torch.nn.Linear(1024, 1024)
    self.softmax = nn.LogSoftmax()
    
    self.attout = torch.nn.Linear(2048, 1024)
    
    
    self.lstm = torch.nn.LSTM(input_size=1324, hidden_size=1024, num_layers=1)
    
    
    self.gen = torch.nn.Linear(23262, 1024)

    self.hidden = Parameter(torch.randn((48, 1, 1024)))
    

  def forward(self, targ, encoder_out):
    self.hidden = Parameter(torch.randn((48, len(encoder_out), 1024)))
    embedded = self.embedding(targ)
    output = embedded
    #print hidden
    #hidden = torch.cat((hidden[0], hidden[1]), 2)[0]
    
    #sc = Variable(torch.zeros((len(encoder_out),)))
    #for i in range(len(encoder_out)):
    #  sc[i] = self.score(encoder_out[i], self.hidden)
    sc = self.score(encoder_out, self.hidden)
    a = self.softmax(sc)#.unsqueeze(2)
    mult = torch.mul(a.unsqueeze(2), encoder_out)
    print mult
    s = torch.sum(mult, 0)
    #print s
    #print s
    context = torch.tanh(self.attout(torch.cat((s, hidden), 2)))
#1 seq 48
#1 48 1024

#48 1024
    output = torch.cat((context, embedded), 2)
    
    output, hiddenN = self.lstm(score, hidden)
    
    generated = self.gen(output)
    return generated, hiddenN
  
  def score(self, h_s, h_t):
    #seqlen = len(h_s)
    h_t = self.attin(h_t)
    #h_t = h_t_.view(48, seqlen, 1024)
    print h_t
    return torch.bmm(h_t, h_s.transpose(0,1))
    #= self.attin(encoder_out)
    #sc = Variable(torch.zeros(48))
    #for i in range(48):
    #  sc[i] = torch.dot(hidden[i], score[i])
    #return self.hidden.dot(score)
    
    
class NMT(nn.Module):
  def __init__(self, vocab_size):
    super(NMT, self).__init__()
    self.Encoder = Encoder(vocab_size)
    self.Decoder = Decoder()
    self.enchidden = None

  def forward(self, input, targ):
    
    encout, self.enchidden = self.Encoder(input)
    
    return self.Decoder.forward(targ, encout)