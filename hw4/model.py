import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np

# TODO: Your implementation goes here
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
    
    #word embedding (vocab_size, embedding_dimension)
    embedding_size = 15
    self.vocab_size = vocab_size
    self.we = Parameter(torch.randn(vocab_size, embedding_size))  # random word embedding
    
    self.hidden_size = 16
    self.hidden = Variable(torch.randn(1, self.hidden_size))

    self.i2h = Parameter(torch.randn(embedding_size + self.hidden_size, self.hidden_size))
    self.i2o = Parameter(torch.randn(embedding_size + self.hidden_size, vocab_size))

  def forward(self, input):
    hidden = Variable(torch.randn(input.data.shape[1], self.hidden_size))
    o = Variable(torch.zeros((input.data.shape[0], input.data.shape[1], self.vocab_size)))
    for i in range(input.data.shape[0]):
      combined = torch.cat((self.we[input.data[i,:],:], hidden), 1)
      #48x31 31x16
      hidden = torch.tanh(torch.mm(combined, self.i2h)) #48x16
      #48x31 31x7000
      output = torch.exp(torch.tanh(torch.mm(combined, self.i2o)))
      output = torch.div(output,torch.sum(output))
      o[i,:,:] = output
    return o


# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass
