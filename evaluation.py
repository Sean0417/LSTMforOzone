import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def evaluation(model,test_x,test_y,lossfunction):
    test_x = torch.from_numpy(test_x)
    test_x = torch.tensor(test_x,dtype=torch.float32).view(-1,1,6)
    model.load_state_dict(torch.load('models/best_model.pkl'))
    # model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval() # prep model for evaluation
    if lossfunction == 'MSE':
        criterion = torch.nn.MSELoss(size_average=True)
    elif lossfunction == 'MAE':
        criterion = torch.nn.L1Loss(size_average=True)
    with torch.no_grad():
        pred_y = model(test_x)

        test_loss = criterion(pred_y,torch.from_numpy(test_y.reshape(-1,1)))
    pred_y = pred_y.view(-1).data.numpy()
    return test_y, pred_y, test_loss