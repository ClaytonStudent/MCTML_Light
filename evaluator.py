import numpy as np
import pandas as pd
import random
import copy
from sklearn.pipeline import Pipeline
from Basic import names,name_parameter,name_model
from sklearn.model_selection import ShuffleSplit,cross_val_score

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
    model = name_model[parameters[0]] # have the model name
    hyper = name_parameter[parameters[0]] # have the hyperparameter space : dictionary
    hyper_constant = {}
    for i,j in enumerate(hyper.keys()):
        hyper_constant[j] = parameters[1:][i]
    model = model.set_params(**hyper_constant)
    score = cross_val_score(model, X, y, cv=5) # Model CV Score
    return score.mean()

def get_score_with_one_feature(parameters):
    model = name_model[parameters[0]] # have the model name
    hyper = name_parameter[parameters[0]] # have the hyperparameter space : dictionary
    hyper_parameters = parameters[-len(hyper):] # have the hyperparameter exact values
    hyper_constant = {}
    for i,j in enumerate(hyper.keys()):
        hyper_constant[j] = hyper_parameters[i]
    model = model.set_params(**hyper_constant)
    feature = parameters[1]
    X_trans = feature.fit_transform(X)
    score = cross_val_score(model, X_trans, y, cv=3) # Model CV Score
    return score.mean()
    
    
    

    
    
#def pipeline_to_score(pipe):
#    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
#    score = cross_val_score(pipe,X,y,cv=cv)
#    return score.mean()

#def pipeline_to_score_with_std(pipe):
#    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
#    score = cross_val_score(pipe,X,y,cv=cv)
    
    #score = score_minius_k_std(score,k=1)  用来计算score = 平均值-K倍标准差 体现pipeline的稳定性
#    return score.mean()
###################################
### 4. selecte index of parameters
##################################
#def get_index_of_zeros(a):
#    return [i for i in range(len(a)) if a[i]=='0']

#def get_index_of_ones(list_name):
#    return [i for i,val in enumerate(list_name) if val==1]

###################################
### 5. used to replace rollout
##################################
#def change_char(s, p):
#    return s[:p]+ '1'+s[p+1:]          
            
#def update_parameter(state):
#    parameter = copy.deepcopy(state.id) 
#    model = state.model
#    if '1' not in model:
#        avaiable_model = [i+state.model_index for i in range(len(state.model))]
#        model_index = random.choice(avaiable_model)
#        parameter = change_char(parameter,model_index)
#    elif '0' not in state.preprocessor:
#        return parameter
#    avaiables_sampler = get_index_of_zeros(state.preprocessor)
#    number = random.randint(0,len(avaiables_sampler))
#    sampler = random.sample(avaiables_sampler,number)
#    for s in sampler:
#        parameter = change_char(parameter,s)
#    return parameter

#def pipeline_construction(parameter):
#    pipelist = []
#    for index,value in enumerate(parameter):
#        if value == '1':
#            clf = search_space_values[index]()
#            clf_name = search_space_names[index]
#            pipelist.append((clf_name,clf))
#    return Pipeline(pipelist)

#def score_minius_k_std(score,k=1):
#    return np.mean(score) - k*np.std(score)   
#def get_index_of_one(list_name):
#    return [i for i,val in enumerate(list_name) if val==1]

#def set_default_parameters(clf):
#    default_parameter = clf.get_hyperparameter_search_space().get_default_configuration().get_dictionary()
#    return clf(**default_parameter,random_state=1)

#def parameter_to_pipeline(parameter,search_space_dic):
#    parameter_index = get_index_of_one(parameter)
#    pipe_list=[]
#    for index in parameter_index:
#        pipe_list.append((str(index),set_default_parameters(search_space_dic[index])))
#    return Pipeline(pipe_list) 

############################
### 3. MCTS policy
############################   
    
    
###################################
### 6. used in tree.py 
##################################
#def change_char(s, p):
#    return s[:p]+ '1'+s[p+1:]

#def get_next_state(state_id,action):
#    return change_char(state_id,action)