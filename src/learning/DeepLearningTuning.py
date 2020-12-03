from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
# Import required libraries
import pandas as pd
import numpy as np 
import sklearn
import time
from sklearn.model_selection import train_test_split


# Keras specific

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from keras.utils import np_utils
# tuning

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice



def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    df = pd.read_csv("fiveS.csv")
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    spec_nums = {"AA":1,"OD":2,"PC":3,"SR":4,"VG":5}
    df["spec_num"] = [spec_nums[i] for i in df["species"]]
    #pre-process
    target_column = ['spec_num'] # response varable
    predictors = list(set(list(df.columns))-set(["Width",
    "Time",'Unnamed: 0',"well","dilution","media","copper","species","spec_num"])) # predictors # predictors
    df[predictors] = df[predictors]/df[predictors].max()# normalise predictors
    df = df.sample(frac=1) #shuffle rows
    # split data
    X = df[predictors].values
    Y = df[target_column].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=40)

    #specify that response is categorical
    Y_train = to_categorical(Y_train)[:,1:]
    Y_test = to_categorical(Y_test)[:,1:]

    return X_train, Y_train, X_test, Y_test


def create_model(X_train, Y_train, X_test, Y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    # define model
    model = Sequential()
    model.add(Dense({{choice([5,6,12,24,48,96,192,384])}}, activation={{choice(['relu', 'sigmoid'])}}, input_dim=12))
    hiddenlayers = {{choice(['one','two','three'])}}
    if hiddenlayers in ['one', 'two']:
        model.add(Dense({{choice([6,12,24,48,96,192,384])}}, activation={{choice(['relu', 'sigmoid'])}}))
    if hiddenlayers == 'two':
        model.add(Dense({{choice([6,12,24,48,96,192,384])}}, activation={{choice(['relu', 'sigmoid'])}}))

    model.add(Dense(5, activation='softmax'))

# Compile the model
    model.compile(optimizer={{choice(['adam', 'sgd', 'rmsprop'])}}, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

# build the model
    if 'results' not in globals():
        global results
        results = []
    start = time.time()
    model.fit(X_train, Y_train, epochs={{choice([10,20,40,80])}})
    #result = model.fit(X_train, Y_train, epochs={{choice([10,20,40,80])}})
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    #print('Test accuracy:', acc)
    #valLoss = result.history['val_mean_absolute_error'][-1]
    parameters = space
    parameters["acc"] = acc
    parameters["time"] = str(int(time.time() - start)) + "sec"
    results.append(parameters)    
    my_df=pd.DataFrame(results)
    my_df.to_csv('params.csv')
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50000,
                                          trials=Trials(),eval_space=True)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)


