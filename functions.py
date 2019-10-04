from Basic import names,name_parameter
import numpy as np
# 把state.path或者sample 换成等长的数值型表示
def convert_state(path):
    if path is None or len(path) <=0:
        print('The state is something wrong!')
        return None
    else:
        state = [-1,-1,-1,-1]
        clf_ind = names.index(path[0])
        state[0] = clf_ind
        params = list(name_parameter[path[0]].values())
        for i in range(len(path[1:])):
            param_index = params[i].index(path[i+1])
            state[i+1] = param_index
    state = [i+1 for i in state]
    return state


# 提取出训练数据 state：Q 和 sample：score
def extract_training_data(path,nodes):
    X = [convert_state(sample[0]) for sample in path] +  [convert_state(node.state.path) for node in nodes if node._Q>0]
    y = [sample[1] for sample in path] + [node._Q for node in nodes if node._Q>0] 
    return np.array(X),np.array(y)

# 检查某一个节点的所有子节点被访问次数
def child_visit_times(node):
    return [i._n_visits for i in node._children.values()]

# find the best parameter and score
def find_best(agent):
    paths = agent.mcts._storage.ps
    scores = [ps[1] for ps in paths]
    best = paths[scores.index(max(scores))]
    return best[0],best[1]


# find the useless hyperparameter and delete the nodes from mcts tree
def find_useless(paths,path): 
    for p in paths:
        if p[1] == path[1] and p[0][0] == path[0][0]:
            clf_hyper = find_one_difference(p[0],path[0])
            if clf_hyper[0] != False and clf_hyper not in useless:
                delete_nodes(clf_hyper)

def find_one_difference(p1,p2):
    k = [True if p1[i]!=p2[i] else False for i in range(1,len(p1))]
    if sum(k) == 1:
        return p1[0],k.index(True)
    else:
        return False,False

def delete_nodes(clf_hyper):
    clf_name = clf_hyper[0]
    index = clf_hyper[1]
    hyper = list(name_parameter[clf_name].keys())[index]
    parent_nodes = [node for node in nodes if node.state.path[0]==clf_name and node.state.level == index+1]
    random_action = random.choice(name_parameter[clf_name][hyper])
    name_parameter[clf_name][hyper] = [random_action]
    for parent in parent_nodes:
        parent._children = {random_action:parent._children[random_action]}
        parent.state.actions = [random_action]
    
