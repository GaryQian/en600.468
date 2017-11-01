import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F

# TODO: Your implementation goes here
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
    
    #word embedding (vocab_size, embedding_dimension)
    embedding_size = 15
    self.vocab_size = vocab_size
    self.we = nn.Linear(vocab_size, embedding_size)  # random word embedding
    
    self.hidden_size = embedding_size
    self.hidden = Variable(torch.randn(embedding_size))

    self.i2h = nn.Linear(embedding_size + embedding_size, embedding_size)
    self.i2o = nn.Linear(embedding_size + embedding_size, vocab_size)
    
    
    self.softmax = nn.LogSoftmax()

  def forward(self, input):
    #input = input concat self.hidden
    
    #self.hidden = torch.tanh(sum(input * weights1))
    
    #output = softmax(sum(self.hidden * weights2))
    
    onehot = np.zeros((len(input), self.vocab_size))
    for i,v in enumerate(input.data.numpy()):
      onehot[i][v] = 1
    for n in range(len(input)):
      #TODO (D L E W I S) Make this more efficient by not using np to make a onehot
      oh = Variable(torch.from_numpy(onehot[n,:]).type(torch.FloatTensor))
      oh = np.zeros((self.vocab_size,))#Variable(torch.zeros((len(input), self.vocab_size)))
      oh = Variable(torch.from_numpy(oh).type(torch.FloatTensor))
      #END TODO
      we = self.we(oh)
      combined = torch.cat((we, self.hidden), 0)
      self.hidden = F.tanh(self.i2h(combined))
      output = F.tanh(self.i2o(combined))
      output = self.softmax(output)
    return output


# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass
