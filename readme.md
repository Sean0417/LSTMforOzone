# LSTM for predicting future Ozone
## Introduction
<p>This is a project using LSTM to predict future Ozone quantity. In this project we use every 6 days' Ozone data to predict the seventh day's Ozone quantity.

## Environments
Python 3.10.9
<P>Visual Studio Code

## Requirements
matplotlib==3.8.0<br>
numpy==1.26.1<br>
pandas==2.1.2<br>
torch==2.1.0<br>
wandb==0.15.12<br>

## Deploy Python virtual Environment
Script in shell:<br>
<ul>
    <li>python3 -m pip install --user virtualenv</li>
    <li>python3 -m venv venv</li>
    <li>source venv_test/bin/activate</li>
    <li>pip install -r requirements.txt</li>
</ul>

## How to run the programm
  open the command window in current folder and enter 
  '''shell
  source start_py.sh
  '''
## File Structure
### main.py<br>
- the main function of the programm where lays the sequence of the whole project<br>
### dataloader.py <br>
- sort_data_by_slidingwindow(): sort the data in sliding window, the return value type are two arrays.
- train_validate_test_data_split(): divide the data into training set, validation set and test set.
- dataPrepare(): get the required dataloader.
### model.py
-  LSTM_Regression: the class of the training model, which consists of a lstm layer and a linear layer.
### train.py
- training_cycle(model,epoch_sum,train_loader,val_loader,learningRate,patience,size_average=True): function used to conduct the training cycle.
- change_best_model_name():change the best model's file name in order to call it in evaluation module.
### early_stopping.py
- used to conduct early stopping in validation step.
### evaluation.py
- evalution(): perform the test process after training.

### start_py.sh
- shell script used to run python project. you can also change the parameters here.
### plot.py
- plot_prediction_curve(): plot the prediction results with the data on test set.
- plot_Train_and_validation_loss():plot the learning curve(training loss and validation loss).
## contact
- email: xinyu.xie@stud.uni-due.de
- telephone: +49 15252305027