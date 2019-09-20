import numpy as np
import pandas as pd
import random
import copy
from sklearn.pipeline import Pipeline
from Basic import names,name_parameter,name_model
from sklearn.model_selection import ShuffleSplit,cross_val_score
from functions import sample_to_clf

def load_dataset(dataset_name):
    print("Loading the dataset:",dataset_name)
    df = pd.read_csv(dataset_name)
    X = np.array(df.iloc[:,:-1])
    y = np.array(df.iloc[:,-1])
    return X,y

df_name = 'dataset_3'
X,y = load_dataset("datasets/"+df_name+".csv")

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
    clf = sample_to_clf(parameters)
    score = cross_val_score(clf, X, y, cv=5)
    return score.mean()