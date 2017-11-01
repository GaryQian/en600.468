import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import math

# TODO: Your implementation goes here
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
    
    #word embedding (vocab_size, embedding_dimension)
    embedding_size = 32
    self.vocab_size = vocab_size
    self.we = Parameter(torch.randn(vocab_size, embedding_size))  # random word embedding
    
    self.hidden_size = 16

    self.i2h = Parameter(torch.randn(embedding_size + self.hidden_size, self.hidden_size))
    self.h2o = Parameter(torch.randn(self.hidden_size, vocab_size))
    self.bias = Variable(torch.zeros((48,self.vocab_size)))
    self.softmax = torch.nn.LogSoftmax()
    self.reset_parameters()

  def to_scalar(self,var):
        # returns a python float
        return var.view(-1).data.tolist()[0]

  def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return self.to_scalar(idx)

  def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

  def forward(self, input):
    input_len, batch_size = input.size()
    hidden = Variable(torch.randn(batch_size, self.hidden_size))
    o = Variable(torch.zeros((input_len, batch_size, self.vocab_size)))
    for i in range(input_len):
      
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      hidden = torch.tanh(torch.mm(combined, self.i2h))
      #print hidden[1,3:6]
      #print self.h2o
      output = self.softmax(torch.add(torch.mm(hidden, self.h2o), self.bias))
      
      #output = torch.add(torch.mm(hidden, self.h2o), self.bias)
      #result = output.sub(self.log_sum_exp(output))#torch.log(torch.sum(torch.exp(output)))
      #print result
      #print self.softmax(torch.add(torch.mm(hidden, self.h2o), self.bias))
      print hidden
      o[i,:,:] = output
    return o
    
  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    i = 0
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)
      i += 1
    print i


# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    #word embedding (vocab_size, embedding_dimension)
    embedding_size = 32
    self.vocab_size = vocab_size
    self.we = Parameter(torch.randn(vocab_size, embedding_size))
    
    self.hidden_size = 16

    self.i2h = Parameter(torch.randn(embedding_size, self.hidden_size))
    self.i2o = Parameter(torch.randn(self.hidden_size, vocab_size))

  def forward(self, input_batch):
    hidden = Variable(torch.randn(input.data.shape[1], self.hidden_size))
    o = Variable(torch.zeros((input.data.shape[0], input.data.shape[1], self.vocab_size)))
    for i in range(input.data.shape[0]):
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      #48x31 31x16
      hidden = torch.tanh(torch.mm(combined, self.i2h)) #48x16
      #48x31 31x7000
      output = torch.mm(hidden, self.i2o)
      output -= torch.log(torch.sum(torch.exp(output)))
      o[i,:,:] = output
    return o