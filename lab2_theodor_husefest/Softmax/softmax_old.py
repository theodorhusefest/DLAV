import numpy as np
from random import shuffle


def one_hot_encode(y):
    y_hot = np.zeros((y.size, y.max() + 1))
    y_hot[np.arange(y.size), y] = 1
    return y_hot

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability.                         #
  #############################################################################

  s = X@W
  y_hot = one_hot_encode(y)
  pred = np.take(s, y)

  softmax = np.exp(pred) / (np.sum(np.exp(s),axis = 1))
  loss = -np.log(softmax)
  loss = np.mean(loss)
  
  print(X.T.dot(y_hot).shape)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW