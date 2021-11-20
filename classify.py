import tensorflow as tf
from util import train_step, test, visualize


def classify(model,optimizer,num_epochs,train_ds,valid_ds):
    """
    Trains and tests our predefined model.
        Args:
            model: our untrained model
            optimizer: optimizer for the model
            num_epochs: number of training epochs
            train_ds: our training dataset
            valid_ds our validation set for testing and regulating hyperparameters
        Results:
            result: list with losses and accuracies
            model: our trained MLP model
    """

    tf.keras.backend.clear_session()

    # Initialize the loss: categorical cross entropy
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()

    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []

    # Testing on our test_ds once before we begin
    model.set_training(False)
    test_loss, test_accuracy = test(model, valid_ds, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # Testing on our train_ds once before we begin
    train_loss, _ = test(model, train_ds, cross_entropy_loss)
    train_losses.append(train_loss)

    # Training our model for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

        # training (and calculating loss while training)
        epoch_loss_agg = []
        model.set_training(True)

        for input,target in train_ds:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))


        # testing our model in each epoch to track accuracy and test loss
        model.set_training(False)
        test_loss, test_accuracy = test(model, valid_ds, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    results = [train_losses,test_losses,test_accuracies]
    return results, model
