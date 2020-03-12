import numpy as np
from random import shuffle


def softmax_loss_vectorized(W, X, y, reg = 0):
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability.                         #
    #############################################################################
    
    N = X.shape[0] 

    # substract max X-value to avoid numerical instability
    X -= np.max(X) 
    
    S = np.matmul(X, W)
    S = np.exp(S)

    ''' calc loss with one hot vector '''
    y_pred = S / np.expand_dims(np.sum(S, axis=1), axis=1)
    
    y_onehot = np.zeros_like(y_pred)
    np.put_along_axis(y_onehot, np.expand_dims(y, axis=1), 1, axis=1)

    loss =- np.sum(y_onehot * np.log(y_pred), axis=1)
    loss_mean = np.mean(loss) + reg*np.sum(np.square(W))



    ''' calc gradient for loss function '''
    dL = (y_pred - y_onehot)/N
    # X  shape: N x 3073
    # dL shape: N x 10
    # dW shape: 3037 x 10 x N
    

    dW = X.T@dL + reg*2*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss_mean, dW
