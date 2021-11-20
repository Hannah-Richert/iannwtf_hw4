import tensorflow as tf
import os
import matplotlib.pyplot as plt
from util import visualize, loading_data, test
from model import MyModel
from classify import classify
from layer import MyDense


if __name__ == "__main__":
    #os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # Hyperparameters
    epochs = 20
    learning_rate = 0.05

    # lists vof visualization
    results = []
    trained_models = []
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    test_accuracies = []

    # getting our data
    train_ds,valid_ds,test_ds = loading_data()

    # training multiple models with different conditions
    for optimizer in [tf.keras.optimizers.SGD(learning_rate),tf.keras.optimizers.Adam(learning_rate)]:
        for model in [MyModel(None,0),MyModel('l1_l2',0),MyModel(None,0.5),MyModel('l1_l2',0.5)]:
            tf.keras.backend.clear_session()
            result,model = classify(model,optimizer,epochs,train_ds,valid_ds)
            results.append(result)
            trained_models.append(model)

    # splitting our results into multiple lists
    for result in results:
        train_losses.append(result[0])
        valid_losses.append(result[1])
        valid_accuracies.append(result[2])
        
    # after adjusting our hyper parameters, checking the models on our unseen test_ds
    for model in trained_models:
        _, accuracy = loss,accuracy = test(model,test_ds,tf.keras.losses.BinaryCrossentropy())
        test_accuracies.append(accuracy.numpy())

    # visualize the losses and accuracies in a grid_plot
    visualize(train_losses,valid_losses,valid_accuracies,test_accuracies)
    plt.show()
