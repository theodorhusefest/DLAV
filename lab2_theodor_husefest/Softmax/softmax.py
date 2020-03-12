import numpy as np
from random import shuffle


def softmax_loss_vectorized(W, X, y):
    """
    Softmax loss function, vectorized version.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability.                         #
    #############################################################################


    # https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    X -= np.max(X) # substract max X-value to avoid exp(too high number)
    S = np.matmul(X, W)
    S = np.exp(S)
    #S_sum = np.sum(S, axis=1)

    ''' calc loss choosing target label '''
    #target = np.choose(y, S.T)  # https://stackoverflow.com/questions/17074422/select-one-element-in-each-row-of-a-numpy-array-by-column-indices
    #loss_1 = -np.log(target / S_sum)
    #loss_mean_1 = np.mean(loss_1, axis=0)

    ''' calc loss with one hot vector '''
    y_pred = S / np.expand_dims(np.sum(S, axis=1), axis=1)

    y_onehot = np.zeros_like(y_pred)
    np.put_along_axis(y_onehot, np.expand_dims(y, axis=1), 1, axis=1)

    # https://medium.com/analytics-vidhya/softmax-classifier-using-tensorflow-on-mnist-dataset-with-sample-code-6538d0783b84
    # https://www.quora.com/Is-the-softmax-loss-the-same-as-the-cross-entropy-loss
    loss_2 = - np.sum(y_onehot * np.log(y_pred), axis=1)
    loss_mean_2 = np.mean(loss_2)



    ''' calc gradient for loss function '''
    # http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html

    dL = y_pred - y_onehot

    #dW = np.matmul(X, dL.T)

    # X  shape: N x 3037
    # dL shape: N x 10
    # dW shape: 3037 x 10 x N
    x = np.expand_dims(X[0,:], axis=1)
    dl = np.expand_dims(dL[0,:], axis=1)

    dW = np.zeros((3073, 10, N))
    for i in range(N):
        x = np.expand_dims(X[i, :], axis=1)
        dl = np.expand_dims(dL[i, :], axis=1)
        dW[:, :, i] = np.matmul(x, dl.T)

    dW = np.mean(dW, axis=2)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss_mean_2, dW
