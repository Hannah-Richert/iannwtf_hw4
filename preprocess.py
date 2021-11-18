
from util import prepare_data
import pandas as pd
import numpy as np
import tensorflow as tf

def loading_data():
    # load the datasets from the given path
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    #df=(df-df.min())/(df.max()-df.min()) #normalizing the df

    df_median =df['quality'].median()
    df['quality'] = np.where(df['quality'] >= df_median, True, False)
    (rows,cols) = df.shape

    inputs = df.drop(['quality'],axis=1)
    targets = df['quality']
    full_ds = tf.data.Dataset.from_tensor_slices((inputs,targets))

    train_ds = full_ds.take(int(0.7*rows))
    remaining = full_ds.skip(int(0.7*rows))
    valid_ds = remaining.take(int(0.15*rows))
    test_ds = remaining.skip(int(0.15*rows))

    # apply preprocessing to the datasets
    train_ds = train_ds.apply(prepare_data)
    test_ds = test_ds.apply(prepare_data)
    valid_ds = valid_ds.apply(prepare_data)

    return train_ds,test_ds,valid_ds
