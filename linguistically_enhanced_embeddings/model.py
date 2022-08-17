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

        self.downscale = nn.Linear(hidden_size, 40)
        self.upscale = nn.Linear(hidden_size, 768)


    def forward(self, input_tensor, input_feature_lengths):
        input_tensor = pack_padded_sequence(input_tensor, input_feature_lengths)
        output, hidden = self.lstm(input_tensor)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = (hidden[0].sum(0, keepdim=True), hidden[1].sum(0, keepdim=True))
        
        hidden_dsc = self.downscale(hidden[0]).squeeze()
        hidden_upsc = self.upscale(hidden[0]).squeeze()
        
        return hidden_dsc, hidden_upsc, output, hidden
