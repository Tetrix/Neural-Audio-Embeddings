import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np



class Encoder(nn.Module):
    def __init__(self, input_tensor, hidden_size, num_layers):
        super(Encoder, self).__init__()

        self.input_tensor = input_tensor
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
                            self.input_tensor,
                            self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=True
                            )


    def forward(self, input_tensor, input_feature_lengths):
        input_tensor = pack_padded_sequence(input_tensor, input_feature_lengths)
        output, hidden = self.lstm(input_tensor)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        return output, hidden


    
# Attention decoder
class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, num_layers, encoder_num_layers):
        super(Decoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers

        self.lstm = nn.LSTM(40,
                            self.decoder_hidden_size,
                            num_layers=self.num_layers,
			    bidirectional=False)
       
        self.out = nn.Linear(self.decoder_hidden_size, 40)

    def forward(self, input_tensor, decoder_hidden):
        decoder_output, decoder_hidden = self.lstm(input_tensor, decoder_hidden)
        
        return decoder_output, decoder_hidden

