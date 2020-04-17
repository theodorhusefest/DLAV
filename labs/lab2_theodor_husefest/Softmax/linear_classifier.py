import numpy as np
from .softmax import *


class LinearClassifier(object):

  def __init__(self):
    self.W = None
    
  def calculate_accuracy():
    pass

  def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=False, reg = 0.0, X_val=None, y_val=None):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      # self.W = 0.001 * np.random.randn(dim, num_classes)
      
      # Initialize using normal distribution 
        self.W = np.random.normal(loc=0.0, scale=0.001, size=(dim, num_classes))

    num_batches = int(np.floor(X.shape[0]/batch_size))
    print('Number of batches: ', num_batches)
    # Run stochastic gradient descent to optimize W
    train_loss_history = []
    val_loss_history = []
    for it in range(num_iters):

      for batch in range(num_batches):


      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      
        ind = np.random.choice(X.shape[0], batch_size, replace = True)
        X_batch = X[ind]
        y_batch = y[ind]
        
        
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
        _, grad = self.loss(X_batch, y_batch, reg)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
        self.W -= learning_rate*grad
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      train_loss, _ = self.loss(X, y, reg)
      train_loss_history.append(train_loss)

      if X_val is not None and y_val is not None:
        val_loss, _ = self.loss(X_val, y_val, reg=0)
        val_loss_history.append(val_loss)

      if verbose and it % 5 == 0:
        print('iteration %03d / %03d: train_loss %.03f / val_loss %.03f' % (it, num_iters, train_loss, val_loss))

    return train_loss_history, val_loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    
    S = np.matmul(X, self.W)
    S = np.exp(S)
    
    y_pred = S / np.expand_dims(np.sum(S, axis=1), axis=1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return np.argmax(y_pred, axis = 1)
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass 


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

