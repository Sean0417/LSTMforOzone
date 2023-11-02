import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def evaluation(model,test_loader,lossfunction):
    # initialization
    labels = []
    predictions = []
    test_loss = 0

    # define which lossfunction to choose
    if lossfunction == 'MSE':
        criterion = torch.nn.MSELoss(size_average=True)
    elif lossfunction == 'MAE':
        criterion = torch.nn.L1Loss(size_average=True)

    # model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval() # prep model for evaluation
    for i, data in enumerate(test_loader): # in test_loader, batch size is 1
        x,y = data
        y = np.array(y.view(-1)) # reshape the labels in order for later criterion
        x = torch.tensor(x, dtype=torch.float32).view(-1,1,6)
        with torch.no_grad():
            y_predict = model(x)
        y_predict = y_predict.view(-1).data.numpy()
        labels.extend(y)
        predictions.extend(y_predict)
        test_loss += criterion(y, y_predict)
    
    test_loss = test_loss/len(y)
    return y, y_predict, test_loss