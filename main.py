import dataloader
import model as md
import time
import train
import evaluation
import argparse
import plot
import numpy as np
import torch
import wandb
def main(args):
    # 1.data preparation
    print(args.is_train)
    x, y = dataloader.sort_data_by_slidingWindow(filepath=args.filepath,col=[8])
    x_train,y_train,x_val,y_val,x_test,y_test = dataloader.train_validate_test_data_split(x_data=x, y_data=y, train_percentage=args.training_percentage, validate_percentage=args.validate_percentage)
    train_loader = dataloader.dataPrepare(x_data=x_train,y_data=y_train, batch_size=args.batch_size, shuffle=True)# train_loader
    val_loader = dataloader.dataPrepare(x_data=x_val,y_data=y_val, batch_size=args.batch_size, shuffle=False) # validate_loader
    test_loader = dataloader.dataPrepare(x_data=x_test,y_data=y_test,batch_size=1,shuffle=False)
    # 2. Model initialization
    model = md.LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size)




    # 3. Model training
    if args.is_train.lower() == "yes":
        model = md.LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size)
        loss_cycle_validation_min = []
        losses_train = []
        losses_val =[]
        for n in range(args.num_exps):
            model = md.LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size) # in every iteration we need to initialize the model in order to start randomly
            model, loss_train, loss_val = train.training_cycle(model=model,
                                                            epoch_sum=args.num_of_epochs,
                                                            train_loader=train_loader,
                                                            val_loader=val_loader,
                                                            patience=args.patience,
                                                            learningRate=args.learning_rate,
                                                            index_of_main_cyle=n)
            loss_cycle_validation_min.append(np.min(loss_val))
            losses_train.append(loss_train)
            losses_val.append(loss_val)
            print("round"+str(n+1)+" has been done")
        index = np.argmin(loss_cycle_validation_min) # argmin get the index of the minimum element of the array
        train.change_best_model_name(index=index,model_filepath="./models")
        loss_train = losses_train[index]
        loss_val = losses_val[index]
        # plot the training and validation loss curve
        plot.plot_Train_and_validation_loss(loss_train=loss_train,loss_val=loss_val)
    else:
        model = md.LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size)
    
    
    
    
    # 4. Model evaluation
    labels, predictions,loss_test = evaluation.evaluation(model=model,test_loader=test_loader,lossfunction=args.lossfunction)
    plot.plot_prediction_curve(y=labels, y_predict=predictions,loss_test=loss_test)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Set hyper parameters for the training.')
    parser.add_argument('--num_exps',type=int,required=True,default=1,help="times of running experiments")
    parser.add_argument('--is_train',type=str,help="parameter to determine whether run training cycle or not")
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

    args = parser.parse_args()
    main(args=args)