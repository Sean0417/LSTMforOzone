import torch
import numpy as np
import wandb

def evaluation(model,test_loader,lossfunction):
    test_loss_sum = 9
    n = 1
    predictions = []
    labels = []
    model.load_state_dict(torch.load('models/best_model.pkl'))
    model.eval()
    if lossfunction == 'MSE':
        criterion = torch.nn.MSELoss(reduction="mean")
    elif lossfunction == 'MAE':
        criterion = torch.nn.L1Loss(size_average=False)
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            x, y = data
            y = torch.tensor(y,dtype=torch.float32).view(-1,1)
            x = torch.tensor(x, dtype=torch.float32).view(-1,1,6)
            y_pred = model(x)

            test_loss = criterion(y_pred,y).item()
            test_loss_sum +=test_loss
            n +=1
            y_pred = y_pred.numpy()
            y = y.data.numpy()
            predictions.extend(y_pred)
            labels.extend(y)
            wandb.log({"test_loss":test_loss})
        test_loss = test_loss_sum/n
    return labels, predictions, test_loss # labels and predictions are lists, each element is a numpy dtype=float32
    