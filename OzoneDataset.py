import torch
from torch.utils.data import Dataset # Dataset is an abstract class, can only be herited
class OzoneDataset(Dataset):
    def __init__(self,x_data,y_data) -> None: # x_data and y_data are numpy
 
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)

    def __len__(self):
        return self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]