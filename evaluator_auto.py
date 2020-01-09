import numpy as np
import pandas as pd
import os
import random
import copy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit,cross_val_score
from search_space import check_list


def state_to_pipeline(state):
    pipe = []
    for i in range(3):
        params = dict(list(map(list, state[i+3])))
        check_list[state[i]].update(params)
        pipe.append((state[i],check_list[state[i]].model))
    return Pipeline(pipe)

def load_dataset(dataset_name):
    df = pd.read_csv(dataset_name)
    #df = df.groupby('class').apply(pd.DataFrame.sample, frac=0.5).reset_index(drop=True)
    X = np.array(df.iloc[:,:-1])
    y = np.array(df.iloc[:,-1])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    return X_train,X_test,y_train,y_test

def policy_value_fn(actions):    
    if actions == [] or actions is None:
        return None
    policy = [(action,1) for action in actions]
    return policy


def get_score(node,df_name):
    #print(df_name)
    X_train,X_test,y_train,y_test = load_dataset(df_name)
    pipe = state_to_pipeline(node._state.state)
    try:
        score = cross_val_score(pipe, X_train,y_train, cv=5).mean()
    except:
        score = -1
    print(round(score,3))
    return score
    

def rollout_policy_fn():
    pass
        
    
    
