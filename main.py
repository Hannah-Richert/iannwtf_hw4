import tensorflow as tf
import os
import matplotlib.pyplot as plt
from util import visualize, loading_data
from model import MyModel
from classify import classify


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    ep = 10
    train_ds,test_ds,valid_ds = loading_data()
    results_1 = classify(MyModel(None),tf.keras.optimizers.SGD(0.1),ep,train_ds,test_ds,valid_ds)
    results_2 = classify(MyModel(None),tf.keras.optimizers.SGD(0.1,momentum=0.01),ep,train_ds,test_ds,valid_ds)
    results_3 = classify(MyModel(None),tf.keras.optimizers.RMSprop(0.1,momentum=0.01),ep,train_ds,test_ds,valid_ds)
    results_4 = classify(MyModel(None),tf.keras.optimizers.Adam(0.1),ep,train_ds,test_ds,valid_ds)
    #results_2 = classify(MyModel('l1'),tf.keras.optimizers.Adam(0.1),ep,train_ds,test_ds,valid_ds)
    #results_3 = classify(MyModel('l2'),tf.keras.optimizers.Adam(0.1),ep,train_ds,test_ds,valid_ds)

    #loss,accuracy validation set
    print(results_1[3:])
    print(results_2[3:])
    print(results_3[3:])
    print(results_4[3:])

    # visualize the results training and testing set
    visualize(results_1[0],results_1[1],results_1[2])
    visualize(results_2[0],results_2[1],results_2[2])
    visualize(results_3[0],results_3[1],results_3[2])
    visualize(results_4[0],results_4[1],results_4[2])
    plt.show()
