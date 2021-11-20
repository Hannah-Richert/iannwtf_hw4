# iannwtf_hw4

+ Data preperation, hyperparameters and optimization techniques:
  - We took **60%** from our data as **training** and each **20%** for **validation and final testing**.
  - We **normalized** the whole data.
  - We made the targets **binary** based on the **median** of all quality ratings.
  - We set the **learning_rate** to **0.05**.
  - We trained and analysed the models for **20 and 200 epochs**.
  - We trained **8 different models** under with different conditions, and analysed them.
  - We used first **SGD** and later the **Adam** as **optimizer** with their standart parameters.
  - We applied **"l1_l2"** kernel **regularization** to all layers.
  - We applied **dropout** between all layers with a rate of **50%**.
  - We plotted all resluts in a grid, seen below.

# Results:

+ 20 training epochs:
  ![20_epochs](https://user-images.githubusercontent.com/93341845/142739267-938a2ad9-3e1a-4d41-a9f8-94630c641a93.png)
+ 200 training epochs:
  ![200_epochs](https://user-images.githubusercontent.com/93341845/142739086-7e495a31-193c-4fb9-a503-fed8ccf9e12d.png)

+ Analyse and Interpretation: 
  - SGD: Stochastic Gradient DEcent does not easily find an optimum. Our loss-surface might be very shallow with only very few valleys. Regularization and a 50% dropout does not help with it. If we would train for more epochs, we might get lucky and find the a local optimum. Then we would get a similar accuracy, as with Adam.
  - Adam seems to **speed up the training process** a lot. But in our case we **get some overfitting** and not the best general accuracy for Adam alone. 
  -'l1_l2' regulization: When combined with Adam we can see an increase of the test dataset accuracy (**generalization of our model**), but very similar training and validation losses.
  - Applying a 50% dropout rate to the model, while using Adam, seems to **solve and avoid our overfitting problem**. This is very logical, because we always drop some parameters which fit the specific data points. In our case we do not get a good test-set accuracy, the model seems to lack generalization.
 - **Final: Adam,'l1_l2',Dropout: When combining all three optimizations, we get a model, which has a very high accuracy for the validation and our testing datasets and avoiding overfitting.** 

