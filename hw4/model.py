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
    self.bias = Variable(torch.ones((1,self.vocab_size)))
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
    bias = self.bias.repeat(batch_size,1)
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
      o[i,:,:] = output
    return o
    
  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)


# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    #word embedding (vocab_size, embedding_dimension)
    embedding_size = 32
    self.vocab_size = vocab_size
    self.we = Parameter(torch.randn(vocab_size, embedding_size))  # random word embedding
    
    self.hidden_size = 16 / 2

    self.i2h = Parameter(torch.randn(embedding_size + self.hidden_size, self.hidden_size))
    self.h2o = Parameter(torch.randn(self.hidden_size * 2, vocab_size))
    self.bias = Variable(torch.ones((48,self.vocab_size)))
    self.softmax = torch.nn.LogSoftmax()
    self.reset_parameters()

  def forward(self, input):
    input_len, batch_size = input.size()
    #Forward
    hidden = Variable(torch.randn(batch_size, self.hidden_size))
    hiddenf = Variable(torch.randn(input_len + 1, batch_size, self.hidden_size))
    hiddenf[0,:,:] = hidden
    for i in range(input_len):
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      hidden = torch.tanh(torch.mm(combined, self.i2h))
      hiddenf[i + 1,:,:] = hidden
      
    #backward
    hidden = Variable(torch.randn(batch_size, self.hidden_size))
    hiddenb = Variable(torch.randn(input_len + 1, batch_size, self.hidden_size))
    hiddenb[input_len:,:] = hidden
    for i in range(input_len)[::-1]:
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      hidden = torch.tanh(torch.mm(combined, self.i2h))
      hiddenb[i,:,:] = hidden
      
    o = Variable(torch.zeros((input_len, batch_size, self.vocab_size)))
    for i in range(input_len):
      hidden = torch.cat((hiddenf[i,:,:], hiddenb[i+1,:,:]),1)
      output = self.softmax(torch.add(torch.mm(hidden, self.h2o), self.bias))
      o[i,:,:] = output
    return o
    
  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)
      
      

class CustRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(CustRNNLM, self).__init__()
    #word embedding (vocab_size, embedding_dimension)
    embedding_size = 80
    self.vocab_size = vocab_size
    self.we = Parameter(torch.randn(vocab_size, embedding_size))  # random word embedding
    self.hidden_size = 40 / 2

    self.i2h = Parameter(torch.randn(embedding_size + self.hidden_size, self.hidden_size))
    self.h2o = Parameter(torch.randn(self.hidden_size * 2, vocab_size))
    self.softmax = torch.nn.LogSoftmax()
    self.hiddenInit = torch.randn(1, self.hidden_size)
    self.reset_parameters()
    self.dropout_rate = 0.2
    self.we.data[1,:] = torch.zeros(embedding_size)
    self.bias = Parameter(torch.ones((1,self.hidden_size)))

  def getDropoutMask(self, dim):
  	return torch.Tensor(np.random.binomial(np.ones(dim, dtype='int64'),1-dropout_percent)) 


  def forward(self, input, do_dropout=False):
    input_len, batch_size = input.size()
    bias = self.bias.repeat(batch_size,1)

    #Forward
    hidden = Variable(self.hiddenInit.repeat(batch_size, 1))
    hiddenf = Variable(torch.randn(input_len + 1, batch_size, self.hidden_size))
    hiddenf[0,:,:] = hidden
    for i in range(input_len):
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      hidden = torch.tanh(torch.add(torch.mm(combined, self.i2h), bias))
      if do_dropout:
      	hidden =torch.mul(self.getDropoutMask(hidden.size()), hidden)
      hiddenf[i + 1,:,:] = hidden
      
    #backward
    hidden = Variable(self.hiddenInit.repeat(batch_size, 1))
    hiddenb = Variable(torch.randn(input_len + 1, batch_size, self.hidden_size))
    hiddenb[input_len:,:] = hidden
    for i in range(input_len)[::-1]:
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      hidden = torch.tanh(torch.add(torch.mm(combined, self.i2h), bias))
      if do_dropout:
      	hidden =torch.mul(self.getDropoutMask(hidden.size()), hidden)
      hiddenb[i,:,:] = hidden
      
    o = Variable(torch.zeros((input_len, batch_size, self.vocab_size)))
    for i in range(input_len):
      hidden = torch.cat((hiddenf[i,:,:], hiddenb[i+1,:,:]),1)
      output = self.softmax(torch.mm(hidden, self.h2o))
      o[i,:,:] = output
    return o
    
  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)