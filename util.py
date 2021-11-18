import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#@tf.function()
def make_binary(input,threshold):
  """

  """
  encoded_input = input
  return encoded_input


def prepare_data(ds):
  """
  Preparing our data for our model.
    Parameters:
      ds: the dataset we want to preprocess
    Returns:
      ds: preprocessed dataset
  """

  # make targets binary
  ds = ds.map(lambda feature, target: (feature, make_binary(target,6)))

  # cast features and targets to float32
  ds = ds.map(lambda feature, target: (tf.cast(feature, tf.float32),tf.cast(target,tf.float32)))
  # cache the elements
  ds = ds.cache()

  # shuffle, batch, prefetch our dataset
  ds = ds.shuffle(100)
  ds = ds.batch(32)
  ds = ds.prefetch(20)

  return ds

#@tf.function
def train_step(model, input, target, loss_function, optimizer):
  """
  Performs a forward and backward pass for  one dataponit of our training set

    Parameters:
      model: our created MLP model
      input:
      target:
      loss_funcion: function we used for calculating our loss
      optimizer: our optimizer used for packpropagation
    Results:
      loss: our calculated loss for the datapoint
    """

  with tf.GradientTape() as tape:

    # forward step
    prediction = model(input)

    # calculating loss
    loss = loss_function(target, prediction)

    # calculaing the gradients
    gradients = tape.gradient(loss, model.trainable_variables)

  # updating weights and biases
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss


def test(model, test_data, loss_function):
  """
  Test our MLP, by going through our testing dataset,
  performing a forward pass and calculating loss and accuracy

    Parameters:
      model: our created MLP model
      test_data: our preprocessed test dataset
      loss_funcion: function we used for calculating our loss
    Results:


  """
  # initializing lists for accuracys and loss
  accuracy_aggregator = []
  loss_aggregator = []

  for (input, target) in test_data:

    # forward step
    prediction = model(input)

    # calculating loss
    loss = loss_function(target, prediction)

    # calculating accuracy
    accuracy =  np.round(target.numpy(),0) == np.round(prediction.numpy(),0)
    accuracy = np.mean(accuracy)

    # add loss and accuracy to the lists
    loss_aggregator.append(loss.numpy())
    accuracy_aggregator.append(np.mean(accuracy))

  # calculate the mean of the loss and accuracy (for this epoch)
  loss = tf.reduce_mean(loss_aggregator)
  accuracy = tf.reduce_mean(accuracy_aggregator)

  return loss, accuracy

def visualize(train_loss,test_loss,accuracy):
  """
  Displays training and testing loss as well as accuracy per epoch each in a line plot.

    Args:
      train_loss = mean training loss per epoch
      test_loss = mean testing loss per epoch
      accuracy = mean accuracy (testing dataset) per epoch
  """

  fig, axes = plt.subplots(2)
  line_1 = axes[0].plot(train_loss,label="  Train Loss")
  line_2 = axes[0].plot(test_loss, label = "  Test Loss")
  line_3 = axes[1].plot(accuracy, label = "  Test Accuracy")
  plt.xlabel("  Training Epoch")
  axes[0].legend()
  axes[1].legend()
