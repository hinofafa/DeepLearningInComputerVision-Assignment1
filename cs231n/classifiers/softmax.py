import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):
    score = X[i].dot(W)
    ## Normalization trick to prevent large denominator
    score -= np.max(score)
    sum_of_exp_score = np.sum(np.exp(score))
    loss += -np.log(np.exp(score[y[i]])/sum_of_exp_score)
    for j in range(num_class):
      ## proportion of score with sum score
      prop_of_sum = np.exp(score[j])/sum_of_exp_score
      if(j == y[i]):
        # subtract all other wrong proportion
        dW[:, j] += (prop_of_sum-1) * X[i]
      else:
        dW[:, j] += prop_of_sum * X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  ## keepdims can broadcast indexing by keeping one dimension as 1
  scores -= np.max(scores, axis=1, keepdims=1)
  ## axis 1 for preserve row sum for calculate prop_of_sum
  sum_of_exp_scores = np.sum(np.exp(scores),axis=1, keepdims=1)
  prop_of_sum = np.exp(scores)/sum_of_exp_scores

  loss = np.sum(-np.log(prop_of_sum[np.array(range(num_train)),y]))
  inverse_case = np.zeros_like(scores)
  ## Inverse case is a scores like matrix but remark those correct score subtract other proportion
  inverse_case[np.array(range(num_train)),y] = 1
  dW = X.T.dot(prop_of_sum - inverse_case)
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

