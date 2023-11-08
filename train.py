import torch
import torch.nn as nn
from early_stopping import EarlyStopping
import numpy as np
import time
import os
import wandb
class opt_and_cri_functions:
    def __init__(self,model,learningRate,size_average=True):
        self.criterion = torch.nn.MSELoss(size_average=size_average)
        self.optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

def training_validation(model,epoch_sum,train_loader,val_loader,learning_rate,patience,exp_index,model_folder_directory,size_average=True):
    time_start = time.time()

    ocfunction = opt_and_cri_functions(model,learning_rate,size_average)
    optimizer = ocfunction.optimizer
    criterion = ocfunction.criterion

    all_batch_train_losses = []
    all_batch_val_losses = []
    all_epoch_train_losses = []
    all_epoch_val_losses = []
    
    early_stopping = EarlyStopping(patience=patience,verbose=True)

    for epoch in range(1,epoch_sum+1):
        # ============training mode==============
        model.train()  # 
        for inputs, targets  in train_loader:
            targets = targets.view(-1,1)
            inputs = inputs.view(-1,1,6)

            predictions = model(inputs)
            optimizer.zero_grad()
            batch_loss = criterion(predictions, targets)
            batch_loss.backward()
            optimizer.step()
            all_batch_train_losses.append(batch_loss.item())

        epoch_train_loss = np.average(all_batch_train_losses)
        # =================validation mode===========================
        model.eval() 
        for j, v_data in enumerate(val_loader):
            inputs, labels = v_data
            labels = torch.tensor(labels,dtype=torch.float32).view(-1,1)
            inputs = torch.tensor(inputs, dtype=torch.float32).view(-1,1,6)

            predictions = model(inputs) # the out put of the network
            batch_loss = criterion(predictions, labels)
            all_batch_val_losses.append(batch_loss.item())
        
        epoch_val_loss = np.average(all_batch_val_losses)

        # print training/validation statistics
        # calculate average loss over an epoch
        wandb.log({"train_loss": epoch_train_loss,"val_loss":epoch_val_loss})
        all_epoch_train_losses.append(epoch_train_loss)
        all_epoch_val_losses.append(epoch_val_loss)

        epoch_len = len(str(epoch_sum))

        print_msg = (f'round:{exp_index+1}:[{epoch:>{epoch_len}}/{epoch_sum::>{epoch_len}}]'+
                     f'train_loss:{epoch_train_loss:.5f}' + ' '
                     f'valid_loss:{epoch_val_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        all_batch_train_losses = []
        all_batch_val_losses = []

        # early _stopping needs the validation loss to check if if has decreased
        # and if it has, it will make a checkpoint of the current model. Note that early stopping will only store the model with the best validation loss in checkpoint.pt
        early_stopping(epoch_val_loss, model)
        # when it reaches the requirements of stopping,early——stop will be set as True
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # time consumed in one experiment
    time_end = time.time()
    duaration  = time_end - time_start
    print("The training took %.2f"%(duaration/60)+ "mins.")
    time_start = time.asctime(time.localtime(time_start))
    time_end = time.asctime(time.localtime(time_end))
    print("The starting time was ", time_start)
    print("The finishing time was ", time_end)


    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))


    if os.path.exists(model_folder_directory) == False:
        os.makedirs(model_folder_directory)
    else:
        pass

    # save the model with the best validation loss
    save_model_time = time.strftime("%Y%m%d_%H%M%S")
    model_name = 'model_params'+save_model_time+"_"+str(exp_index+1)
    torch.save(model.state_dict(),model_folder_directory+'/'+model_name+'.pkl')


    return model_name, model, all_epoch_train_losses, all_epoch_val_losses