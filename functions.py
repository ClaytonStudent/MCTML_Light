import copy
import random
import numpy as np
from search_space import check_list
# 检查某一个节点的所有子节点被访问次数
def child_visit_times(node):
    return [i._n_visits for i in node._children.values()]

# find the best parameter and score
def find_best(agent):
    paths = agent.mcts._storage.ps
    if len(paths) < 1:
	    return "No",0
    scores = [ps[1] for ps in paths]
    best = paths[scores.index(max(scores))]
    return best[0],best[1]

def save_score(name,k,score):
    print('Saving best score to ',name)
    text_file = open("Storage_2/"+name+".txt", "a")
    text_file.write("Min:{}, Score:{}".format(k,score) + '\n')
    text_file.close()

def save_model(name,params,score):
    print('Saving best model to ',name)
    text_file = open("Storage_2/"+name+".txt", "a")
    text_file.write("Name:{}".format(name) + '\n' + 
                    "Best Parameter: {} ".format(params) + '\n' + 
                    "Best Score: {}".format(score) + '\n' + '\n' + '\n')
    text_file.close()
    
    
    
def game_end(avaiables_actions):
    if avaiables_actions == None:
        return True
    else:
        return False
    
    
# related with progressive windowing
def has_continues(node):
    if node._state.depth >=3 and window(node._n_visits) and node._n_visits >10:
        selected_method = check_list[node._state.state[node._state.depth-3]]
        is_continue,action = generate_random_sample(selected_method.hyper_parameters)
        if is_continue:
            node.add_new_node(action)

def generate_random_sample(params):
    is_continue = False
    params = copy.deepcopy(params)
    for i,j in params.items():
        if type(j)==tuple and type(j[0])== float:
            is_continue = True
            k = random.uniform(j[0],j[1])
        elif type(j)==tuple and type(j[0])== int:
            is_continue = True
            k = int(random.sample(range(j[0],j[1]),1)[0])
        else:
            k = random.choice(j)
        params[i] = k
    return is_continue,tuple((p,v) for p,v in params.items())

def window(N):
    C = 4
    alpha = 0.3
    if int(C*(N+1)**alpha) > int(C*(N)**alpha):
        return True
    else:
        return False
    
 
    
# related with NN
def convert_state(state):
    X = [0,0,0,0,0,0]
    for i in range(3):
        X[i] = list(check_list.keys()).index(state[i]) 
        if  state[i+3] in check_list[state[i]].search_space:
            X[i+3] = check_list[state[i]].search_space.index(state[i+3])
        else:
            X[i+3] = len(check_list[state[i]].search_space) + random.randint(0,10)
    return X

def extract_training_data(ps):
    X = [convert_state(sample[0].state) for sample in ps]
    y = [sample[1] for sample in ps]
    return np.array(X),np.array(y)

