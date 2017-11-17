import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import math
import torch.nn.functional as F

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
  
    def score(self, h_s, h_t):
        h_t = self.attin(h_t)
        return h_t.unsqueeze(1).bmm(h_s.unsqueeze(1).transpose(1,2))

class Decoder(nn.Module):
  def _init_state(self, encoder_hidden):
    """ Initialize the encoder hidden state. """
    if encoder_hidden is None:
      return None
    if isinstance(encoder_hidden, tuple):
      encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
    else:
      encoder_hidden = self._cat_directions(encoder_hidden)
    return encoder_hidden
    
  def _cat_directions(self, h):
    """ If the encoder is bidirectional, do the following transformation.
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
    """
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

  def __init__(self, src_vocab_size, trg_vocab_size):
    super(Decoder, self).__init__()
    
    self.hidden_size = 1024
    self.output_size = trg_vocab_size
    
    self.initParams = torch.load(open("model.param", "rb"))
    
    self.embedding = torch.nn.Embedding(trg_vocab_size, 300)
    
    
    self.attn = Attn('general', self.hidden_size)
    
    self.attin = torch.nn.Linear(self.hidden_size, self.hidden_size)
    self.softmax = nn.LogSoftmax()
    
    self.attout = torch.nn.Linear(2048, self.hidden_size)
    
    
    self.lstm = torch.nn.LSTM(input_size=300, hidden_size=self.hidden_size, num_layers=1)
    
    #23262
    self.gen = torch.nn.Linear(self.hidden_size,trg_vocab_size)
    self.hidden = None#Parameter(torch.randn((48, 1024)))
    self.prev = Variable(torch.randn((48, self.hidden_size)))
    
    self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
    

  def forward(self, targ, encoder_out, encoder_hidden):
    embedded = self.embedding(targ)
    
    decoder_hidden = self._init_state(encoder_hidden)

    output, hidden = self.lstm(embedded, decoder_hidden)

    context = encoder_out.transpose(0,1)
    output = output.transpose(0,1)
    batch_size = output.size(0)
    hidden_size = output.size(2)
    input_size = context.size(1)
    
    #output, attn = self.attention(output, encoder_outputs)
    # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
    attn = torch.bmm(output, context.transpose(1, 2))
    attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)

    # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
    mix = torch.bmm(attn, context)

    # concat -> (batch, out_len, 2*dim)
    combined = torch.cat((mix, output), dim=2)
    # output -> (batch, out_len, dim)
    output = F.tanh(self.attout(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
    
    #print output
    predicted_softmax = F.log_softmax(self.gen(output.view(-1, self.hidden_size))).view(batch_size, self.output_size, -1)
    return predicted_softmax#, hidden, attn
    

    #self.prev, _= self.lstm(output, None)#Variable(torch.zeros((2, 48, 1024,))))
    #generated = self.gen(self.prev)
    #self.prev = self.prev[-1]
    #return generated

  #def score(self, h_s, h_t):
  #  h_t = self.attin(h_t)
  #  res= h_t.unsqueeze(1).bmm(h_s.unsqueeze(1).transpose(1,2))
  #  return res
    
class NMT(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size):
    super(NMT, self).__init__()
    self.Encoder = Encoder(src_vocab_size)
    self.Decoder = Decoder(src_vocab_size, trg_vocab_size)

  def forward(self, input, targ):
    
    encout, hidden = self.Encoder(input)
    
    return self.Decoder(targ, encout, hidden)