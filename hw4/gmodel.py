import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import math

class GRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(GRNNLM, self).__init__()
    #word embedding (vocab_size, embedding_dimension)
    embedding_size = 128
    self.vocab_size = vocab_size
    self.we = Parameter(torch.randn(vocab_size, embedding_size).cuda())  # random word embedding
    self.hidden_size = 64 / 2

    self.i2h1 = Parameter(torch.randn(embedding_size + self.hidden_size, self.hidden_size).cuda())
    self.i2h2 = Parameter(torch.randn(embedding_size + self.hidden_size, self.hidden_size).cuda())
    self.h2o = Parameter(torch.randn(self.hidden_size * 2, vocab_size).cuda())
    self.softmax = torch.nn.LogSoftmax().cuda()
    self.hiddenInit = torch.randn(1, self.hidden_size).cuda()
    self.reset_parameters()
    self.dropout_rate = 0.1
    self.we.data[1,:] = torch.zeros(embedding_size).cuda()
    self.bias = Parameter(torch.ones((1,self.hidden_size)).cuda())

  def getDropoutMask(self, dim):
    return torch.Tensor(np.random.binomial(np.ones(dim, dtype='int64'),1-self.dropout_rate)).cuda() 


  def forward(self, input, do_dropout=False):
    input_len, batch_size = input.size()
    bias = self.bias.repeat(batch_size,1)

    #Forward
    hidden = Variable(self.hiddenInit.repeat(batch_size, 1).cuda())
    hiddenf = Variable(torch.randn(input_len + 1, batch_size, self.hidden_size).cuda())
    hiddenf[0,:,:] = hidden
    for i in range(input_len):
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      hidden = torch.tanh(torch.add(torch.mm(combined, self.i2h1), bias))
      if do_dropout:
        hidden =torch.mul(Variable(self.getDropoutMask(hidden.size()).cuda()), hidden)
      hiddenf[i + 1,:,:] = hidden
      
    #backward
    hidden = Variable(self.hiddenInit.repeat(batch_size, 1).cuda())
    hiddenb = Variable(torch.randn(input_len + 1, batch_size, self.hidden_size).cuda())
    hiddenb[input_len:,:] = hidden
    for i in range(input_len)[::-1]:
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      hidden = torch.tanh(torch.add(torch.mm(combined, self.i2h2), bias))
      if do_dropout:
        hidden =torch.mul(Variable(self.getDropoutMask(hidden.size()).cuda()), hidden)
      hiddenb[i,:,:] = hidden
      
    o = Variable(torch.zeros((input_len, batch_size, self.vocab_size)).cuda())
    for i in range(input_len):
      hidden = torch.cat((hiddenf[i,:,:], hiddenb[i+1,:,:]),1)
      output = self.softmax(torch.mm(hidden, self.h2o))
      o[i,:,:] = output
    return o
    
  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)
