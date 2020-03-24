# import the necessary packages
import data_utils as du
import argparse
import numpy as np
import torch
from models import ConvNet, ConvNet_Type2

#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################


def predict_usingCNN(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    # Note: For the predictions, you have to return the index of the max    #
    # value                                                                 #
    #########################################################################

    # load network
    fp_checkpoint = 'best_model.ckpt'
    net = ConvNet_Type2()
    net.load_state_dict(torch.load(fp_checkpoint))

    # init dataloader
    x = torch.from_numpy(X).float()
    test_dataset = torch.utils.data.TensorDataset(x)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # evaluate network
    net.eval()
    y_pred = np.empty(0)
    with torch.no_grad():
        for data in test_dataloader:
            images = data[0]
            y_pred_batch = torch.argmax(net.predict(images), dim=1).numpy()
            y_pred = np.append(y_pred, y_pred_batch)
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred
   

def main(filename, group_number):
    X, Y = du.load_CIFAR_batch(filename)

    X, Y = X[:100, :, :, :], Y[:100]  # TODO: remove, only for debugging to avoid memory overflow

    mean_pytorch = np.array([0.4914, 0.4822, 0.4465])
    std_pytorch = np.array([0.2023, 0.1994, 0.2010])
    X_pytorch = np.divide(np.subtract( X/255 , mean_pytorch[:,np.newaxis,np.newaxis]), std_pytorch[:,np.newaxis,np.newaxis])
    prediction_cnn = predict_usingCNN(X)
    acc_cnn = sum(prediction_cnn == Y)/len(X)
    print("Group %s ... CNN= %f" % (group_number, acc_cnn))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", default="data/cifar-10-batches-py/test_batch", required=False, help="path to test file")
    ap.add_argument("-g", "--group", default="Theodor and Simon", required=False, help="group number")
    args = vars(ap.parse_args())
    main(args["test"], args["group"])
