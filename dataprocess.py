import numpy as np
from OzoneDataset import OzoneDataset
from torch.utils.data import DataLoader # DataLoader is a class to help us loading data in Pytorch
import pandas as pd
import torch

def sort_data(filepath,col):
        odata = pd.read_csv(filepath,usecols=col)
        odata = np.array(odata)
        odata_len = odata.shape[0]

        X = []
        y = []

        for i in range(odata_len-6):
            _x = odata[i:(i+6),0]

            X.append(_x)
            y.append(odata[i+6,0])

        X = (np.array(X)).astype(np.float32)
        y = (np.array(y)).astype(np.float32)
        
        return X, y

def data_split_normalization(x_data, y_data, train_percentage, validate_percentage):
     train_size = int(len(x_data)*train_percentage)
     validate_size = int(len(x_data)*validate_percentage)

     train_x, train_y = x_data[:train_size],y_data[:train_size]
     validate_x, validate_y = x_data[train_size:train_size+validate_size],y_data[train_size:train_size+validate_size]
     test_x, test_y = x_data[train_size+validate_size:],y_data[train_size+validate_size:]

     min_train_x, max_train_x = get_min_max(train_x)# in test and validation set, also use the max and min in train
     norm_train_x = min_max_normalize(x=train_x,max_x=max_train_x,min_x=min_train_x)
     min_train_y, max_train_y = get_min_max(train_y)
     norm_train_y = min_max_normalize(x=train_y,max_x=max_train_y,min_x=min_train_y)
     
     min_val_x, max_val_x = get_min_max(validate_x)
     norm_val_x = min_max_normalize(x=validate_x,max_x=max_val_x,min_x=min_val_x)
     min_val_y, max_val_y = get_min_max(validate_y)
     norm_val_y = min_max_normalize(x=validate_y,max_x=max_val_y,min_x=min_val_y)

     norm_test_x = min_max_normalize(x=test_x,max_x=max_train_x,min_x=min_train_x)
     norm_test_y = min_max_normalize(x=test_y,max_x=max_train_y,min_x=min_train_y)

     # return train_x, train_y, validate_x, validate_y, test_x, test_y # np.array
     return norm_train_x, norm_train_y, norm_val_x, norm_val_y, norm_test_x, norm_test_y

def get_min_max(x):
      min_x = np.min(x)
      max_x = np.max(x)
      return min_x, max_x
def min_max_normalize(x,max_x,min_x):
      norm_x = (x - min_x)/(max_x)
      return norm_x

def prepare_dataloader(x_data,y_data,batch_size,shuffle=True,num_workers=2):
     dataset = OzoneDataset(x_data=x_data,y_data=y_data)

     dataLoader = DataLoader(
          dataset=dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          num_workers=num_workers
     )

     return dataLoader





