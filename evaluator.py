import numpy as np
import pandas as pd
import os
import random
import copy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from Basic import names,name_parameter,name_model
from sklearn.model_selection import ShuffleSplit,cross_val_score
from functions import convert_state
from network import import_model,predict
fname = 'save/model.h5'

def load_dataset(dataset_name):
    print("Loading the dataset:",dataset_name)
    df = pd.read_csv(dataset_name)
    df = df.groupby('class').apply(pd.DataFrame.sample, frac=0.5).reset_index(drop=True)
    X = np.array(df.iloc[:,:-1])
    y = np.array(df.iloc[:,-1])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    return X_train,y_train

df_name = 'dataset_22'
X,y = load_dataset("datasets/"+df_name+".csv")
#X,y = load_dataset("datasets/CC18/"+df_name+"_m.csv")

def policy_value_fn(actions):    
    if actions == [] or actions is None:
        return None
    prob = 1/len(actions)
    policy = [(action,prob) for action in actions]
    return policy

def rollout_policy_fn(actions):
    action_probs = np.random.rand(len(actions))
    return zip(actions, action_probs)

def get_score(parameters):
    #### NN predicted score
    #X_test = convert_state(parameters)
    #X_test = np.array((X_test))
    #nn_model = import_model(fname)
    #nn_predict = predict(nn_model,X_test)
    #print('The predicted value: ',nn_predict)
    
    model = name_model[parameters[0]] # have the model name
    hyper = name_parameter[parameters[0]] # have the hyperparameter space : dictionary
    hyper_constant = {}
    for i,j in enumerate(hyper.keys()):
        hyper_constant[j] = parameters[1:][i]
    model = model.set_params(**hyper_constant)
    score = cross_val_score(model, X, y, cv=5) # Model CV Score
    return score.mean()