from dataprocess import sort_data
from dataprocess import data_split
from dataprocess import prepare_dataloader
from model import LSTM_Regression
from train import training_validation
from evaluation import evaluation
import argparse
from plot import plot_learning_curve
from plot import plot_prediction_curve
import numpy as np
import wandb
import os

def main(args):
    # data preparation
    x, y = sort_data(filepath=args.filepath,col=[8])
    x_train,y_train,x_val,y_val,x_test,y_test = data_split(x_data=x, y_data=y, train_percentage=args.training_percentage, validate_percentage=args.validate_percentage)

    train_loader = prepare_dataloader(x_data=x_train,y_data=y_train, batch_size=args.batch_size, shuffle=True)# train_loader
    val_loader = prepare_dataloader(x_data=x_val,y_data=y_val, batch_size=args.batch_size, shuffle=False) # validate_loader
    test_loader = prepare_dataloader(x_data=x_test,y_data=y_test,batch_size=1,shuffle=False)

    # wandb initialization
    wandb.init(project='LSTMpredictOzone',
            job_type="training",
            reinit=True,
            )
    wandb.watch(model,log="all")


    # Model training
    if args.is_train == True:
        print("===================train and validation====================")
        # experiments loop
        for exp_idx in range(args.num_exps):
            # model initialization
            model = LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size) # in every iteration we need to initialize the model in order to start randomly
            model_name, model, all_epoch_train_loss, all_epoch_val_loss = training_validation(model=model,
                                                            epoch_sum=args.num_of_epochs,
                                                            train_loader=train_loader,
                                                            val_loader=val_loader,
                                                            patience=args.patience,
                                                            learning_rate=args.learning_rate,
                                                            exp_index=exp_idx,
                                                            model_folder_directory=args.model_folder_dir)
            # plot the learning curve
            plot_learning_curve(train_loss=all_epoch_train_loss,val_loss=all_epoch_val_loss,plot_folder_dir=args.plot_folder_dir,model_name=model_name)

            print("round"+str(exp_idx+1)+" has been done")
        

    
    else:
        # model initialization
        model = LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size)
        
    
    
    
    # Model evaluation
    # first set the best loss to infinity
    best_test_loss= float('inf')
    best_test_loss_model=''
    targets_best = []
    predictions_best = []
    
    print("============test================")

    for file in os.listdir(args.model_folder_dir):
        targets, predictions, test_loss = evaluation(model=model,test_loader=test_loader,lossfunction=args.lossfunction, model_filepath=args.model_folder_dir+'/'+file)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_loss_model = file.split('.')[0]
            targets_best = targets
            predictions_best =predictions
        else:
            pass

    plot_prediction_curve(y=targets_best, y_predict=predictions_best,test_loss=best_test_loss,plot_folder_dir=args.plot_folder_dir)
    print("The best model with least test loss so far in these experiments is "+best_test_loss_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Set hyper parameters for the training.')
    parser.add_argument('--num_exps',type=int,required=True,default=1,help="times of running experiments")
    parser.add_argument('--is_train',action="store_true",help="parameter to determine whether run training cycle or not")
    parser.add_argument('--filepath',type=str, required=True,help='file directory')
    parser.add_argument('-lr','--learning_rate',type=float,default=1e-2,required=True,help='learning rate')
    parser.add_argument('-tp','--training_percentage',type=float,default=0.7,required=True,help='the percentage of the training sets')
    parser.add_argument('-vp','--validate_percentage',type=float,default=0.1,required=True,help='the percentage of the validationset')
    parser.add_argument('-bs','--batch_size',type=int,default=40,required=True,help='batch size')
    parser.add_argument('-is','--input_size',type=int,default=6,required=True,help='input size')
    parser.add_argument('-hs','--hidden_size',type=int,required=True,help='size of the hidden layers')
    parser.add_argument('--patience',type=int, default=10,required=True, help='patience of early Stopping')
    parser.add_argument('-es','--num_of_epochs',type=int,default=100,required=True,help = 'the sum of the epochs')
    parser.add_argument('--lossfunction',type=str, required=True, default='MSE',help="The type of the loss function.")
    parser.add_argument('--model_folder_dir',type=str, required=True)
    parser.add_argument('--test_model_path',type=str,help="when the experiments is only for testing, you need to assign which model to initialize")
    parser.add_argument('--plot_folder_dir',type=str,required=True,help="the folder directory where plot results are stored")
    args = parser.parse_args()
    main(args=args)