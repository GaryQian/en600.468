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

    
class Attn(nn.Module):
  def __init__(self, method, hidden_size):
      super(Attn, self).__init__()
      
      self.method = method
      self.hidden_size = hidden_size
      
      self.softmax = nn.LogSoftmax()
      
      self.attn = nn.Linear(self.hidden_size, hidden_size)

  def forward(self, hidden, encoder_outputs):
    max_len = encoder_outputs.size(0)
    this_batch_size = encoder_outputs.size(1)

    # Create variable to store attention energies
    attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

    # For each batch of encoder outputs
    for b in range(this_batch_size):
      # Calculate energy for each encoder output
      for i in range(max_len):
        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

    # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
    return self.softmax(attn_energies).unsqueeze(1)
  
  def score(self, hidden, encoder_output):
    energy = self.attn(encoder_output)
    energy = hidden.dot(energy)
    return energy

class Decoder(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size):
    super(Decoder, self).__init__()
    
    hidden_size = 1024
    
    self.initParams = torch.load(open("model.param", "rb"))
    
    self.embedding = torch.nn.Embedding(trg_vocab_size, 300)
    
    
    self.attn = Attn('general', 1024)
    
    self.attin = torch.nn.Linear(1024, 1024)
    self.softmax = nn.LogSoftmax()
    
    self.attout = torch.nn.Linear(2048, 1024)
    
    
    self.lstm = torch.nn.LSTM(input_size=1324, hidden_size=1024, num_layers=1)
    
    #23262
    self.gen = torch.nn.Linear(1024,trg_vocab_size)
    self.hidden = None#Parameter(torch.randn((48, 1024)))
    self.prev = Variable(torch.randn((48, 1024)))
    
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    

  def forward(self, targ, encoder_out):
    embedded = self.embedding(targ)
    output = embedded
    
    #sc = Variable(torch.zeros((len(encoder_out),48,)))
    #for i in range(len(encoder_out)):
    #  sc[i] = self.score(encoder_out[i], self.hidden)
    #a = self.softmax(sc)
    #a = a.repeat(1024,1,1).transpose(0,1).transpose(1,2)
    ##broadcasting
    #mult = torch.mul(a, encoder_out)
    #s = torch.sum(mult, 0)
    #
    #context = torch.tanh(self.attout(torch.cat((s, self.hidden), 1)))
    #output = torch.cat((context.repeat(len(embedded), 1,1), embedded), 2
    
    sc = Variable(torch.zeros((len(encoder_out),48,)))
    for i in range(len(encoder_out)):
      sc[i] = self.score(encoder_out[i], self.prev)
    a = self.softmax(sc)
    context = a.bmm(encoder_out.transpose(0, 1))
    
    prev = self.prev.squeeze(0) # S=1 x B x N -> B x N
    context = context.squeeze(1)       # B x S=1 x N -> B x N
    concat_input = torch.cat((prev, context), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    
    #a = a.repeat(1024,1,1).transpose(0,1).transpose(1,2)
    #broadcasting
    #mult = torch.mul(encoder_out, a.unsqueeze(2).repeat(1, 1, 1024))
    #s = torch.sum(mult, 0)
    #
    #context = torch.tanh(self.attout(torch.cat((s, self.prev), 1)))
    #output = torch.cat((context.repeat(len(embedded), 1,1), embedded), 2)
    
    #--------------------------------
    
    #output, hiddenN = self.lstm(embedded, self.hidden)
    
    ## Calculate attention from current RNN state and all encoder outputs;
    ## apply to encoder outputs to get weighted average
    #attn_weights = self.attn(output, encoder_outputs)
    #context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N
    #
    ## Attentional vector using the RNN hidden state and context vector
    ## concatenated together (Luong eq. 5)
    #rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
    #context = context.squeeze(1)       # B x S=1 x N -> B x N
    #concat_input = torch.cat((rnn_output, context), 1)
    #concat_output = F.tanh(self.concat(concat_input))
    
    
    self.prev, self.hidden = self.lstm(concat_out, self.hidden)#Variable(torch.zeros((2, 48, 1024,))))
    generated = self.gen(self.prev)
    return generated
  
  def score(self, h_s, h_t):
    h_t = self.attin(h_t)
    print h_s
    return h_t.unsqueeze(1).bmm(h_s.unsqueeze(1).transpose(1,2))
    #a = Variable(torch.zeros((48,1)))
    #for i in range(len(h_t)):
    #    a[i] = torch.dot(h_s[i], h_t[i])
    #return a.transpose(0,1)

    
    
class NMT(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size):
    super(NMT, self).__init__()
    self.Encoder = Encoder(src_vocab_size)
    self.Decoder = Decoder(src_vocab_size, trg_vocab_size)

  def forward(self, input, targ):
    
    encout, hidden = self.Encoder(input)
    
    return self.Decoder(targ, encout)