import matplotlib.pyplot as plt
import numpy as np
import os
def plot_Train_and_validation_loss(train_loss, valid_loss):
    # visualize the loss as the network trained
    plt.figure()
    plt.plot(range(1,len(train_loss)+1),train_loss, label= 'Train Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find postion of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1
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
    plt.close()


def plot_prediction_curve(labels, y_predict, test_loss):
    plt.figure()
    # plt.plot(y_predict,'r',label = 'prediction')
    plt.plot(labels[500:600], 'b', label='ground truth')
    plt.plot(y_predict[500:600],'r',label = 'prediction')
    plt.title('Ozone predictions with test loss='+str(test_loss.data.numpy()))
    plt.xlabel('time')
    plt.ylabel('Ozone')
    plt.xticks(np.arange(0, 100, step = 10))
    plt.legend(loc='best')
    # plt.savefig('pic/result.png',format='png',dpi=200)
    if os.path.exists('./pic'):
        plt.savefig('pic/result.png',format='png',dpi=200)
    else:
        os.makedirs("./pic")
        plt.savefig('pic/result.png',format='png',dpi=200)
    plt.close() 