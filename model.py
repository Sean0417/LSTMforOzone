import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader # DataLoader is a class to help us loading data in Pytorch

class LSTM_Regression(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, num_layers = 1):
        super(LSTM_Regression,self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x,_ = self.lstm(_x) #(the output of x is seq*batch*hidden_size)
        s,b,h = x.shape
        x = x.view(s*b,h)
        x = self.fc(x)
        x.view(s,b,-1)
        return x