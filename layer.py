import tensorflow as tf
import numpy as np

class MyDense(tf.keras.layers.Layer):
  """
  Our own custom layer function, which inherits from the keras.layers.Layer class
    Functions:
      init: constructor
      call: calculates output tensor for this layer
      build: creates weights and bias when call is the first time run
  """

  def __init__(self, units, kernel_regularizer = None,dropout = .5, activation=tf.nn.softmax):
    """
    Constructs an fully connected layer.
      Args:
        units: perceptrons of our layer (int)
        kernel_regularizer: Regularization method applied to our layer
        dropout: dropout rate applied to our layer (float)
        activation: activation function of our layer
    """

    super(MyDense, self).__init__( )

    self.units = units
    self.activation = activation
    self.kernel_regularizer = kernel_regularizer
    self.dropout = dropout
    # save if we are training or testing the model
    self.is_training = True



  def call(self, inputs):
    """
    Calculates the output of our layer. (forwads_step)
      Args:
        inputs: input tensor of our layer
      Returns:
        x: the output of our layer
    """

    # applying dropout (dropping some previous outputs),only  when training!
    if self.is_training:
        # creating binominal distribution in shape of the input
        dropout_dist = np.random.binomial(1,(1-self.dropout), size=tf.shape(inputs))
        # by applying the distribution, some values get dropped
        inputs *= dropout_dist

    x = tf.matmul(inputs, self.w) + self.b
    x = self.activation(x)

    return x


  def set_training2(self,is_training):
      """
      Setter for the training variable.
      """
      self.is_training = is_training


  def build(self,input_shape):
    """
    Creates random weights and bias from a normal distribution for our layer.
      Args:
        input_shape: dimension of our input-tensor
    """

    self.w = self.add_weight(shape=(input_shape[-1],self.units),
                             initializer='random_normal',
                             regularizer=self.kernel_regularizer,
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                            initializer='random_normal',
                            regularizer=self.kernel_regularizer,
                            trainable=True)
