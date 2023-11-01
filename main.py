import dataloader
import model as md
import time
import train
import evaluation
import argparse
import plot
import numpy as np
import torch
def main(args):
    # 1.data preparation
    print(args.is_train)
    original_x, original_y = dataloader.sort_data_by_slidingWindow(filepath=args.filepath,col=[8])
    train_x,train_y,val_x,val_y,test_x,test_y = dataloader.train_validate_test_data_split(x_data=original_x,y_data=original_y,train_percentage=args.training_percentage,validate_percentage=args.validate_percentage)
    train_loader = dataloader.dataPrepare(x_data=train_x,y_data=train_y,batch_size=args.batch_size,shuffle=True)# 训练集
    val_loader = dataloader.dataPrepare(x_data=val_x,y_data=val_y,batch_size=args.batch_size,shuffle=False) # 测试集
    # test_loader = dataloader.dataPrepare(x_data=test_x,y_data=test_y, batch_size=1, shuffle=False) # 测试集


    # 2. Model building
    model = md.LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size)
    # 3. Model training
    if args.is_train.lower() == "yes":
        model = md.LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size)
        cycle_validation_min_loss = []
        train_losses = []
        val_losses =[]
        for n in range(args.cycle):
            model = md.LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size) # in every iteration we need to initialize the model in order to start randomly
            model, train_loss, val_loss = train.training_cycle(model=model,
                                                            epoch_sum=args.num_of_epochs,
                                                            train_loader=train_loader,
                                                            val_loader=val_loader,
                                                            patience=args.patience,
                                                            learningRate=args.learning_rate,
                                                            index_of_main_cyle=n)
            cycle_validation_min_loss.append(np.min(val_loss))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print("round"+str(n+1)+" has been done")
        index = np.argmin(cycle_validation_min_loss) # argmin get the index of the minimum element of the array
        train.change_best_model_name(index=index,model_filepath="./models")
        train_loss = train_losses[index]
        val_loss = val_losses[index]
        # plot the training and validation loss curve
        plot.plot_Train_and_validation_loss(train_loss=train_loss,valid_loss=val_loss)
    else:
        model = md.LSTM_Regression(input_size=args.input_size,hidden_size=args.hidden_size)
    # 4. Model evaluation
    labels, y_predict,test_loss = evaluation.evaluation(model=model,test_x=test_x,test_y=test_y,lossfunction=args.lossfunction)

    plot.plot_prediction_curve(labels=labels, y_predict=y_predict,test_loss=test_loss)
    print("The test loss of this model is "+ test_loss)




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Set hyper parameters for the training.')
    parser.add_argument('--cycle',type=int,required=True,default=1,help="sum of times of running training cycle")
    parser.add_argument('--is_train',type=str,required=True,help="parameter to determine whether run training cycle or not")
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