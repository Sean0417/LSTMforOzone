# LSTM for predicting future Ozone
## Introduction
<p>This is a project using LSTM to predict future Ozone quantity. In this project we use every 6 days' Ozone data to predict the seventh day's Ozone quantity.

## Environments
Python 3.10.9
<P>Visual Studio Code

## Requirements
contourpy==1.1.1<br>
cycler==0.12.1<br>
filelock==3.12.4<br>
fonttools==4.43.1<br>
fsspec==2023.10.0<br>
Jinja2==3.1.2<br>
kiwisolver==1.4.5<br>
MarkupSafe==2.1.3<br>
matplotlib==3.8.0<br>
mpmath==1.3.0<br>
networkx==3.2<br>
numpy==1.26.1<br>
packaging==23.2<br>
pandas==2.1.2<br>
Pillow==10.1.0<br>
pyparsing==3.1.1<br>
python-dateutil==2.8.2<br>
pytz==2023.3.post1<br>
six==1.16.0<br>
sympy==1.12<br>
torch==2.1.0<br>
typing_extensions==4.8.0<br>
tzdata==2023.3<br>

## Deploy Python virtual Environment
Script in shell:<br>
'''shell
python3 -m pip install --user virtualenv<br>
python3 -m venv venv_test<br>
source venv_test/bin/activate<br>
pip install -r requirements.txt
'''
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
- 
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