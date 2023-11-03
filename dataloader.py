import numpy as np
import torch
from torch.utils.data import Dataset # Dataset is an abstract class, can only be herited
from torch.utils.data import DataLoader # DataLoader is a class to help us loading data in Pytorch
import pandas as pd

def sort_data_by_slidingWindow(filepath,col):
        odata = pd.read_csv(filepath,usecols=col)
        odata = np.array(odata)
        odata_len = odata.shape[0]

        x_data = []
        y_data = []

        for i in range(odata_len-6):
            _x = odata[i:(i+6),0]

            x_data.append(_x)
            # _y = odata[i+6]
            y_data.append(odata[i+6,0])

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data

def train_validate_test_data_split(x_data, y_data, train_percentage, validate_percentage):
     train_size = int(len(x_data)*train_percentage)
     validate_size = int(len(x_data)*validate_percentage)
     train_x, train_y = x_data[:train_size],y_data[:train_size]
     validate_x, validate_y = x_data[train_size:train_size+validate_size],y_data[train_size:train_size+validate_size]
     test_x, test_y = x_data[train_size+validate_size:],y_data[train_size+validate_size:]
     return train_x, train_y, validate_x, validate_y, test_x, test_y # np.array

class OzoneDataset(Dataset):
    def __init__(self,x_data,y_data) -> None:
 
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)

    def __len__(self):
        return self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
def dataPrepare(x_data,y_data,batch_size,shuffle=True,num_workers=2):
     dataset = OzoneDataset(x_data=x_data,y_data=y_data)
     dataLoader = DataLoader(
          dataset=dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          num_workers=num_workers
     )
     return dataLoader


def get_real_y(filepath): # return value type is numpy
        odata = pd.read_csv(filepath,usecols=[8])
        odata = np.array(odata)
        odata_len = len(odata)
        return odata,odata_len


