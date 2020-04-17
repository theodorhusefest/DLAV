# import the necessary packages
import Softmax.data_utils as du
import argparse
import numpy as np
from Softmax.linear_classifier import Softmax
from Pytorch.Net import Net
import torch

#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

def predict_usingPytorch(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################
    
    net = Net()
    checkpoint = torch.load("./Pytorch/model.ckpt")
    net.load_state_dict(checkpoint)
    pred = np.empty(0)

    net.eval()
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(F.softmax(outputs).data, 1)
            pred = np.append(pred,predicted.numpy())
            
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return pred.astype(int)

def predict_usingSoftmax(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################
    best_softmax = Softmax()
    with open('./Softmax/softmax_weights_acc_0.400.pkl') as f:
        best_softmax.W = pickle.load(f)
        
    y_pred = best_softmax.predict(X)
    
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred

def main(filename, group_number):

    X,Y = du.load_CIFAR_batch(filename)

    ### Modified this part
    mean_pytorch = np.array([0.4914, 0.4822, 0.4465])
    std_pytorch = np.array([0.2023, 0.1994, 0.2010])
    X_pytorch = np.divide(np.subtract(X, mean_pytorch[np.newaxis, np.newaxis, :]), std_pytorch[np.newaxis, np.newaxis, :])
    prediction_pytorch = predict_usingPytorch(torch.Tensor(np.moveaxis(X_pytorch, -1, 1)))

    X = np.reshape(X, (X.shape[0], -1))
    mean_image = np.mean(X, axis = 0)
    X -= mean_image
    prediction_pytorch = predict_usingPytorch(X)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    prediction_softmax = predict_usingSoftmax(X)
    acc_softmax = sum(prediction_softmax == Y)/len(X)
    acc_pytorch = sum(prediction_pytorch == Y)/len(X)
    print("Group %s ... Softmax= %f ... Pytorch= %f"%(group_number, acc_softmax, acc_pytorch))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"],args["group"])