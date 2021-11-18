import tensorflow as tf


class MyModel(tf.keras.Model):
  """
  Our own custon MLP model, which inherits from the keras.Model class

    Functions:
      init: constructor of our model
      call: performs forward pass of our model
  """
  def __init__(self,regulization):
    """
    Constructs our MLP model with three fully connected layers.
    """

    super(MyModel, self).__init__()

    # two hidden layers with each 256 perceptrons and sigmoid as activation function
    self.dense_h1 = tf.keras.layers.Dense(16, kernel_regularizer=regulization,activation=tf.nn.sigmoid)
    self.dense_h2 = tf.keras.layers.Dense(8, kernel_regularizer=regulization,activation=tf.nn.sigmoid)
    #self.dense_h3 = tf.keras.layers.Dense(2, kernel_regularizer=regulization,activation=tf.nn.sigmoid)

    # our output layer with 10 perceptrons (our output categories) and softmax activation
    self.dense_o = tf.keras.layers.Dense(1, kernel_regularizer=None,activation=tf.nn.sigmoid)

  def call(self, inputs):
    """
    Performs a forward step in our MLP

      Args:
        inputs: our preprocessed input data, we send through our model
      Results:
        output: the predicted output of our input data
    """
    # first hidden layer
    x = self.dense_h1(inputs)

    # second hidden layer
    x = self.dense_h2(x)
    #x = self.dense_h3(x)

    # output layer
    output = self.dense_o(x)

    return output

class MyModelRegu(tf.keras.Model):
  """
  Our own custon MLP model, which inherits from the keras.Model class

    Functions:
      init: constructor of our model
      call: performs forward pass of our model
  """
  def __init__(self):
    """
    Constructs our MLP model with three fully connected layers.
    """

    super(MyModelRegu, self).__init__()

    # two hidden layers with each 256 perceptrons and sigmoid as activation function
    self.dense_h1 = tf.keras.layers.Dense(32,kernel_regularizer='l1', activation=tf.nn.sigmoid)
    self.dense_h2 = tf.keras.layers.Dense(16,kernel_regularizer='l1', activation=tf.nn.sigmoid)

    # our output layer with 10 perceptrons (our output categories) and softmax activation
    self.dense_o = tf.keras.layers.Dense(1,kernel_regularizer='l1', activation=tf.nn.sigmoid)

  def call(self, inputs):
    """
    Performs a forward step in our MLP

      Args:
        inputs: our preprocessed input data, we send through our model
      Results:
        output: the predicted output of our input data
    """
    # first hidden layer
    x = self.dense_h1(inputs)

    # second hidden layer
    x = self.dense_h2(x)

    # output layer
    output = self.dense_o(x)

    return output
