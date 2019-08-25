import numpy as np
import pandas as pd
import random
#from random import choice
import copy
from Code.setup import search_space_values,search_space_names,search_space_dic



#######################
## 1. Load Dataset
#######################
def load_dataset(dataset_name):
    print("Loading the dataset")
    df = pd.read_csv(dataset_name)
    X = np.array(df.iloc[:,:-1])
    y = np.array(df.iloc[:,-1])
    return X,y
X,y = load_dataset("Datasets/dataset_38_sick_le.csv")
X = X[:1000,:]
y = y[:1000]

###########################
### 2. Set up Pipeline
###########################
from sklearn.pipeline import Pipeline
def get_index_of_one(list_name):
    return [i for i,val in enumerate(list_name) if val==1]

def set_default_parameters(clf):
    default_parameter = clf.get_hyperparameter_search_space().get_default_configuration().get_dictionary()
    return clf(**default_parameter,random_state=1)

def parameter_to_pipeline(parameter,search_space_dic):
    parameter_index = get_index_of_one(parameter)
    pipe_list=[]
    for index in parameter_index:
        pipe_list.append((str(index),set_default_parameters(search_space_dic[index])))
    return Pipeline(pipe_list) 

############################
### 3. MCTS policy
############################    
def policy_value_fn(actions):    
    if actions == [] or actions is None:
        return None
    prob = 1/len(actions)
    policy = [(action,prob) for action in actions]
    return policy


def rollout_policy_fn(actions):
    action_probs = np.random.rand(len(actions))
    return zip(actions, action_probs)


###################################
### 4. selecte index of parameters
##################################
def get_index_of_zeros(a):
    return [i for i in range(len(a)) if a[i]=='0']

def get_index_of_ones(list_name):
    return [i for i,val in enumerate(list_name) if val==1]

###################################
### 5. used to replace rollout
##################################
def change_char(s, p):
    return s[:p]+ '1'+s[p+1:]          
            
def update_parameter(state):
    parameter = copy.deepcopy(state.id) 
    model = state.model
    if '1' not in model:
        avaiable_model = [i+state.model_index for i in range(len(state.model))]
        model_index = random.choice(avaiable_model)
        parameter = change_char(parameter,model_index)
    elif '0' not in state.preprocessor:
        return parameter
    avaiables_sampler = get_index_of_zeros(state.preprocessor)
    number = random.randint(0,len(avaiables_sampler))
    sampler = random.sample(avaiables_sampler,number)
    for s in sampler:
        parameter = change_char(parameter,s)
    return parameter



from sklearn.pipeline import Pipeline
def pipeline_construction(parameter):
    pipelist = []
    for index,value in enumerate(parameter):
        if value == '1':
            clf = search_space_values[index]()
            clf_name = search_space_names[index]
            pipelist.append((clf_name,clf))
    return Pipeline(pipelist)


from sklearn.model_selection import ShuffleSplit,cross_val_score
def pipeline_to_score(pipe):
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    score = cross_val_score(pipe,X,y,cv=cv)
    return score.mean()
    
    
###################################
### 6. used in tree.py 
##################################
def change_char(s, p):
    return s[:p]+ '1'+s[p+1:]

def get_next_state(state_id,action):
    return change_char(state_id,action)