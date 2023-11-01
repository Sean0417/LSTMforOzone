import torch
import torch.nn as nn
from early_stopping import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import time
import os
class opt_and_cri_functions:
    def __init__(self,model,learningRate,size_average=True):
        self.criterion = torch.nn.MSELoss(size_average=size_average)
        self.optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

def training_cycle(model,epoch_sum,train_loader,val_loader,learningRate,patience,index_of_main_cyle,size_average=True):
    time_start = time.time()
    ocfunction = opt_and_cri_functions(model,learningRate,size_average)
    optimizer = ocfunction.optimizer
    criterion = ocfunction.criterion

    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    
    early_stopping = EarlyStopping(patience=patience,verbose=True)

    for epoch in range(1,epoch_sum+1):
        # ============training mode==============
        model.train()  # 
        # mini-Batch
        for i, data in enumerate(train_loader,1):
            inputs, labels = data
            labels = torch.tensor(labels,dtype=torch.float32).view(-1,1)
            inputs = torch.tensor(inputs, dtype=torch.float32).view(-1,1,6)

            y_pred = model(inputs)
            optimizer.zero_grad()
            batch_loss = criterion(y_pred, labels)
            batch_loss.backward()
            optimizer.step()
            train_losses.append(batch_loss.item())

        # ============================================
        # validation mode mode
        model.eval() 

        # mini_batch
        for j, v_data in enumerate(val_loader):
            inputs, labels = v_data
            labels = torch.tensor(labels,dtype=torch.float32).view(-1,1)
            inputs = torch.tensor(inputs, dtype=torch.float32).view(-1,1,6)

            y_pred = model(inputs) # the out put of the network
            batch_loss = criterion(y_pred, labels)
            val_losses.append(batch_loss.item())
        

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        val_loss = np.average(val_losses)
        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)

        epoch_len = len(str(epoch_sum))

        print_msg = (f'round:{index_of_main_cyle+1}:[{epoch:>{epoch_len}}/{epoch_sum::>{epoch_len}}]'+
                     f'train_loss:{train_loss:.5f}' + ' '
                     f'valid_loss:{val_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        val_losses = []

        # early _stopping needs the validation loss to check if if has decreased
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)
        # when it reaches the requirements of stopping,early——stop will be set as True
        if early_stopping.early_stop:
            print("Early stopping")
            break
    time_end = time.time()
    duaration  = time_end - time_start
    print("The training took %.2f"%(duaration/60)+ "mins.")
    time_start = time.asctime(time.localtime(time_start))
    time_end = time.asctime(time.localtime(time_end))
    print("The starting time was ", time_start)
    print("The finishing time was ", time_end)
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    if os.path.exists('./models') == False:
        os.makedirs('./models')
    else:
        pass
    torch.save(model.state_dict(),'models/model_params'+str(index_of_main_cyle)+'.pkl')
    return model, avg_train_losses, avg_val_losses

def change_best_model_name(index,model_filepath): # rename the document of the best model, in order of modularization
    suffix = '.pkl'
    for file in os.listdir(model_filepath):
        if file.split('.')[0] == "model_params"+str(index):
            new_name = file.replace(file,"best_model"+suffix)
            os.rename(os.path.join(model_filepath, file), os.path.join(model_filepath, new_name))
            break