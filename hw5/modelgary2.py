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

    self.embedding = torch.nn.Embedding(vocab_size, 300).cuda()

    self.lstm = torch.nn.LSTM(input_size=300, hidden_size=512, num_layers=1, bidirectional=True).cuda()


  def forward(self, input):
    embedded = self.embedding(input)
    output = embedded
    output, hidden = self.lstm(output, None)
    return output, hidden

class Decoder(nn.Module):
  def _init_state(self, encoder_hidden):
    if encoder_hidden is None:
      return None
    if isinstance(encoder_hidden, tuple):
      encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
    else:
      encoder_hidden = self._cat_directions(encoder_hidden)
    return encoder_hidden
    
  def _cat_directions(self, h):
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

  def __init__(self, src_vocab_size, trg_vocab_size):
    super(Decoder, self).__init__()
    
    self.hidden_size = 1024
    self.output_size = trg_vocab_size
    
    #self.initParams = torch.load(open("model.param", "rb"))
    
    self.embedding = torch.nn.Embedding(trg_vocab_size, 300).cuda()
    
    self.attin = torch.nn.Linear(self.hidden_size, self.hidden_size).cuda()
    
    self.attout = torch.nn.Linear(2048, self.hidden_size).cuda()
    
    
    self.lstm = torch.nn.LSTM(input_size=300, hidden_size=self.hidden_size, num_layers=1).cuda()
    
    #23262
    self.gen = torch.nn.Linear(self.hidden_size,trg_vocab_size).cuda()
    

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

    
class NMT(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size):
    super(NMT, self).__init__()
    self.Encoder = Encoder(src_vocab_size)
    self.Decoder = Decoder(src_vocab_size, trg_vocab_size)

  def forward(self, input, targ):
    
    encout, hidden = self.Encoder(input)
    
    return self.Decoder(targ, encout, hidden)
