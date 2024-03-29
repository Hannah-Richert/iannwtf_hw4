import tensorflow as tf
from util import train_step, test


def classify(model,optimizer,num_epochs,train_ds,valid_ds):
    """
    Trains and tests our predefined model.
        Args:
            model: our untrained model (MyModel object)
            optimizer: optimizer for the model (keras function)
            num_epochs: number of training epochs (int)
            train_ds: our training dataset (set of tuples with tensors)
            valid_ds our validation set for testing and regulating hyperparameters (set of tuples with tensors)
        Results:
            result: list with losses and accuracies (list containing numerical lists)
            model: our trained MLP model (MyModel object)
    """

    tf.keras.backend.clear_session()

    # initialize the loss: categorical cross entropy
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()

    # initialize lists for later visualization.
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    # testing on our valid_ds once before we begin
    model.set_training(False)
    valid_loss, valid_accuracy = test(model, valid_ds, cross_entropy_loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    # Testing on our train_ds once before we begin
    train_loss, _ = test(model, train_ds, cross_entropy_loss)
    train_losses.append(train_loss)

    # training our model for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {valid_accuracies[-1]}')

        # training (and calculating loss while training)
        epoch_loss_agg = []
        model.set_training(True)

        for input,target in train_ds:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))


        # testing our model in each epoch to track accuracy and loss on the validation set
        model.set_training(False)
        valid_loss, valid_accuracy = test(model, valid_ds, cross_entropy_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

    results = [train_losses,valid_losses,valid_accuracies]
    return results, model
