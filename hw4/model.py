import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np

import torch.tanh

# TODO: Your implementation goes here
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
    
    #word embedding (vocab_size, embedding_dimension)
    embedding_size = 15
    self.vocab_size = vocab_size
    self.we = Parameter(torch.randn(vocab_size, embedding_size))  # random word embedding
    
    self.hidden_size = embedding_size
    self.hidden = Parameter(torch.randn(embedding_size))

    self.i2h = Parameter(torch.randn(embedding_size + self.hidden_size, self.hidden_size))
    self.i2o = Parameter(torch.randn(embedding_size + self.hidden_size, vocab_size))

  def forward(self, input):
    for emb in self.we[input,:]:
      combined = torch.cat((emb, self.hidden), 0)
      self.hidden = torch.tanh(torch.mm(combined, self.i2h))
      output = torch.exp(torch.tanh(torch.mm(combined, self.i2o)))
      output = torch.div(output,torch.sum(output))
    return output


# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass
