import tensorflow as tf
from layer import MyDense


class MyModel(tf.keras.Model):
  """
  Our own custon MLP model, which inherits from the keras.Model class

    Functions:
      init: constructor of our model
      call: performs forward pass of our model
  """

  def __init__(self,kernel_regularizer=None,dropout=0):
    """
    Constructs our MLP model with three dense layers.

        Args:
            kernel_regularizer: our regularizer, which will be applied to all layers
            dropout: dropout_rate for intermediate outputs between the layers
    """

    super(MyModel, self).__init__()

    self.dense_h1 = MyDense(32, kernel_regularizer,dropout,activation=tf.nn.sigmoid)
    self.dense_h2 = MyDense(8, kernel_regularizer,dropout,activation=tf.nn.sigmoid)
    self.dense_out = MyDense(1,kernel_regularizer,0,activation=tf.nn.sigmoid)

  def set_training(self, is_training):
      """
      Setter for the training variables of our layers.
      """

      self.dense_h1.set_training2(is_training)
      self.dense_h2.set_training2(is_training)
      self.dense_out.set_training2(is_training)

  def call(self, inputs):
    """
    Performs a forward step in our MLP

      Args:
        inputs: our preprocessed input data, we send through our model
      Results:
        output: the predicted output of our input data
    """
    
    x = self.dense_h1(inputs)
    x = self.dense_h2(x)
    output = self.dense_out(x)

    return output
