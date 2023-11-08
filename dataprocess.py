import numpy as np
from OzoneDataset import OzoneDataset
from torch.utils.data import DataLoader # DataLoader is a class to help us loading data in Pytorch
import pandas as pd

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

def data_split(x_data, y_data, train_percentage, validate_percentage):
     train_size = int(len(x_data)*train_percentage)
     validate_size = int(len(x_data)*validate_percentage)

     train_x, train_y = x_data[:train_size],y_data[:train_size]
     validate_x, validate_y = x_data[train_size:train_size+validate_size],y_data[train_size:train_size+validate_size]
     test_x, test_y = x_data[train_size+validate_size:],y_data[train_size+validate_size:]

     return train_x, train_y, validate_x, validate_y, test_x, test_y # np.array

    
def prepare_dataloader(x_data,y_data,batch_size,shuffle=True,num_workers=2):
     dataset = OzoneDataset(x_data=x_data,y_data=y_data)
     dataLoader = DataLoader(
          dataset=dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          num_workers=num_workers
     )
     return dataLoader





