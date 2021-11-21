# iannwtf_hw4

+ Data preperation, hyperparameters and optimization techniques:
  - We took **60%** from our data as **training** and each **20%** for **validation and final testing**.
  - We **normalized** the dataset (the pandas dataframe).
  - We made the targets **binary** based on the **median** of all quality ratings.
  - We set the **learning_rate** to **0.05**.
  - We trained and analysed the models for **20 and 200 epochs**.
  - We trained **8 different models** under with different conditions, and analysed them.
  - We used first **SGD** and later the **Adam** as **optimizer** with their standart parameters.
  - We applied **"l1_l2"** kernel **regularization** to all layers (with standart parameters).
  - We applied **dropout** between all layers with a rate of **50%**.
  - We plotted all resluts in a grid, seen below.

BE AWARE: Our dataset has only 1599 dataponts, it is very small. We tested all models with the exact same shuffled dataset, but instaltiated our layers with random weights and biases. Therefore the results are not steady and can vary.

# Results:

+ 20 training epochs:
![20_epochs_079167](https://user-images.githubusercontent.com/93341845/142773423-f82e2141-3220-4e0b-a58c-5f4f6c42c196.png)


+ 200 training epochs:
![200_epochs_076823](https://user-images.githubusercontent.com/93341845/142773883-0bb26569-b330-4889-852a-def345d16d75.png)




+ Accuracy of the testing dataset for our final model (Adam_l1-l2_drop-0.5): 20 epochs: 0.79167 / 200 epochs: 0.76823**
+ Analyse and Interpretation: 
  - SGD: Stochastic Gradient decent **does not easily find an optimum**. Our loss-surface might be very shallow with only very few valleys. Regularization and a 50% dropout does not help with it. If we would train for more epochs, we could get lucky and find the a local optimum. Then we would get a similar accuracy, as with Adam.
  - Adam seems to **speed up the training process** a lot. But in our case we **got some overfitting**.
  - Elastic net regulization: When combined with Adam it does not solve our overfitting problem. But it slightly increasees our accuracy for the validation set.
  - Applying a 50% dropout rate to the model, while using Adam, slightly decreases our accuracy for the validation set. But it **decreases our overfitting problem**. This is very logical, because we always drop some parameters which fit the specific data points. 
 - **Final Model - Adam, 'l1_l2' and Dropout: When combining all three optimizations, we get a model, which has a very high accuracy on our validation set and avoids overfitting. We tested the final model on our unseen and unused test_ds and received a high accuracies as well ( run with 20 epochs: 0.79167 / 200 epochs: 0.76823)**

Fazit: The greatest impact on the training process had the optimizer Adam, but Dropout and 'l1_l2' kernel regulization helped with the finetuning and decreased overfitting.

