# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:35:50 2023

@author: rahul
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(torch.nn.Module):
    
    def __init__(self, input_dimension, output_dimension, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        
        self.num_classes = output_dimension
        self.num_layers = num_layers
        self.input_size = input_dimension
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    
    def forward(self, tensors_x):
    
        # Propagate input through LSTM : Add Sliding Windows
        
        #tensors_x = torch.stack([x[i:(i+self.seq_length)] for i in range(len(x)-self.seq_length-1)])
        
        # print("The size of the tensors_x is ===", tensors_x.size())
        
        h_0 = Variable(torch.zeros(
            self.num_layers, tensors_x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, tensors_x.size(0), self.hidden_size))
        
        ula, (h_out, _) = self.lstm(tensors_x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        output = self.fc(h_out)
        
        return output
    