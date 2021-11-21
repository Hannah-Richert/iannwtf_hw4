import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def loading_data():
    """
    Loading and preprocessing the data.

        Results:
            train_ds,test_ds,valid_ds: the preprocessed datasets
            median: median of our dataframe
    """

    # load the data into a pandas dataframe from the given path
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

    # normalizing the cols of my dataframe
    df = (df-df.mean())/df.std()

    # get the shape of the dataframe
    (rows,cols) = df.shape

    # shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # split ip into input and target and create a combined tensor
    inputs = df.drop(['quality'],axis=1)
    targets = df['quality']
    full_ds = tf.data.Dataset.from_tensor_slices((inputs,targets))

    # split the dataset into training, validation and testing dataset
    train_ds = full_ds.take(int(0.6*rows))
    remaining = full_ds.skip(int(0.6*rows))
    valid_ds = remaining.take(int(0.2*rows))
    test_ds = remaining.skip(int(0.2*rows))

    # apply preprocessing to the datasets
    train_ds = train_ds.apply(prepare_data)
    test_ds = test_ds.apply(prepare_data)
    valid_ds = valid_ds.apply(prepare_data)

    return train_ds,valid_ds,test_ds,

@tf.function()
def make_binary(input,threshold):
  """
  Makes our numerical tensor, binary on the basis of our threshhold.
    Args:
        input: input tensor (single number)
        threshold: our threshold
    Results:
        output: an binary tensor
  """

  if input >= threshold:
      output = tf.constant([1])
  else:
      output = tf.constant([0])

  return output


def prepare_data(ds):
  """
  Preparing our data for our model.

    Args:
      ds: the dataset we want to preprocess
    Results:
      ds: preprocessed dataset
  """

  # make the targets binary, threshhold is the median of our dataset
  ds = ds.map(lambda feature, target: (feature, make_binary(target,0.4507)))

  # cast features and targets to float32
  ds = ds.map(lambda feature, target: (tf.cast(feature, tf.float32),tf.cast(target,tf.float32)))

  # cache the elements
  ds = ds.cache()

  # shuffle, batch, prefetch our dataset
  ds = ds.shuffle(200)
  ds = ds.batch(64)
  ds = ds.prefetch(20)
  return ds


def train_step(model, input, target, loss_function, optimizer):
  """
  Performs a forward and backward pass for  one dataponit of our training set

    Args:
      model: our created MLP model
      input: our input
      target: our target
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

    Args:
      model: our created MLP model
      test_data: our preprocessed test dataset
      loss_funcion: function we used for calculating our loss
    Results:
        loss: our mean loss for this epoch
        accuracy: our mean accuracy for this epoch
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


def visualize(train_losses,valid_losses,valid_accuracies, test_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.

    Args:
      train_losses = mean training losses per epoch
      valid_losses = mean testing losses per epoch
      valid_accuracies = mean accuracies (testing dataset) per epoch
      test_accuracies = single mean accuracy for our unseen test_ds
    """

    titles = ["SGD","SGD_l1-l2","SGD_drop-0.5","SGD_l1-l2_drop-0.5","Adam","Adam_l1-l2","Adam_drop-0.5","Adam_l1-l2_drop-0.5",]
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(13, 6)

    # making a grid with subplots
    for i in range(2):
        for j in range(4):
            axs[i,j].plot(train_losses[i*4+j])
            axs[i,j].plot(valid_losses[i*4+j])
            axs[i,j].plot(valid_accuracies[i*4+j])
            axs[i,j].sharex(axs[0,0])
            axs[i,j].set_title(titles[i*4+j]+" \nTestSet Accuracy: "+str(round(test_accuracies[i*4+j],4)))

    fig.legend([" Train_ds loss"," Valid_ds loss"," Accuracy"])
    plt.xlabel("Training epoch")
    fig.tight_layout()
