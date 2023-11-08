import torch
import numpy as np
import wandb
import sys

def evaluation(model,test_loader,lossfunction,filepath):

    test_loss_sum = 9
    predictions = []
    targets = []
    model.load_state_dict(torch.load(filepath))
   
    model.eval()

    if lossfunction == 'MSE':
        criterion = torch.nn.MSELoss(reduction="mean")
    elif lossfunction == 'MAE':
        criterion = torch.nn.L1Loss(size_average=False)
    else:
        print("Please select between MSE and MAE")
        sys.exit(1)

    with torch.no_grad():
        for i,data in enumerate(test_loader):
            x, y = data
            y = y.view(-1,1)
            x = x.view(-1,1,6)
            y_pred = model(x)

            test_loss = criterion(y_pred,y).item()
            test_loss_sum +=test_loss
            y_pred = y_pred.numpy()
            y = y.data.numpy()
            predictions.extend(y_pred)
            targets.extend(y)

        avg_test_loss = test_loss_sum/len(targets)

    return targets, predictions, avg_test_loss # labels and predictions are lists, each element is a numpy dtype=float32
    