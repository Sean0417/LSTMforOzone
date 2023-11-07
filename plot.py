import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
def plot_learning_curve(loss_train, loss_val):
    # visualize the loss as the network trained
    plt.figure()
    plt.plot(range(1,len(loss_train)+1),loss_train, label= 'Train Loss')
    plt.plot(range(1,len(loss_val)+1),loss_val,label='Validation Loss')

    # find postion of lowest validation loss
    minposs = loss_val.index(min(loss_val))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.title("Learning_curve")
    plt.legend(loc='best')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.grid(True)
    if os.path.exists('./pic'):
        plt.savefig('pic/loss.png',format='png',dpi= 200)
    else:
        os.makedirs("./pic")
        plt.savefig('pic/loss.png',format='png',dpi= 200)
    wandb.log({"best_train_validation_curve":wandb.Plotly(plt.gcf())})
    plt.close()


def plot_prediction_curve(y, y_predict, loss_test):
    plt.figure()
    plt.plot(y[500:600], 'b', label='ground truth')
    plt.plot(y_predict[500:600],'r',label = 'prediction')
    plt.title('Ozone predictions with test loss='+str(loss_test))
    plt.xlabel('time')
    plt.ylabel('Ozone')
    plt.xticks(np.arange(0, 100, step = 10))
    plt.legend(loc='best')
    if os.path.exists('./pic'):
        plt.savefig('pic/result.png',format='png',dpi=200)
    else:
        os.makedirs("./pic")
        plt.savefig('pic/result.png',format='png',dpi=200)

    wandb.log({"plot_prediction_curve":wandb.Plotly(plt.gcf())})
    plt.close()