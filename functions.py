from Basic import names,name_parameter,name_model
import numpy as np

# 把搜集到的sample path 转换成可以训练的clf并更新超参数
def sample_to_clf(path):
    if path is None or len(path) <=0:
        return None
    else:
        clf = name_model[path[0]]
        params = name_parameter[path[0]]
        clf_param = {}
        for p in path[1:]:
            clf_param[p[0]] = p[1]
        clf.set_params(**clf_param)
    return clf

# 转换state 到可训练的X，y值
def convert_state(path):
    if path is None or len(path) <=0:
        return None
    else:
        state = [-1,-1,-1,-1]
        clf_ind = names.index(path[0])
        state[0] = clf_ind
        params = name_parameter[path[0]]
        for i in range(len(path[1:])):
            param_name = path[i+1][0] 
            param_value = path[i+1][1]
            state[list(params.keys()).index(param_name)+1] = params[param_name].index(param_value)
        state = [i+1 for i in state]
        return state

def child_been_visited(node):
    if not node.is_leaf():
        Q_values = [n._Q for n in list(node._children.values())]
        return min(Q_values)>0
    else:
        return False

def child_probs(node):
    Q_values = [n._Q for n in list(node._children.values())]
    Q_probs = [q/sum(Q_values) for q in Q_values]
    return Q_probs

def extract_training_data(nodes):
    X = [convert_state(node.state.path) for node in nodes if child_been_visited(node)]
    y = [child_probs(node) for node in nodes if child_been_visited(node)] 
    return np.array(X),np.array(y)




'''
### For Fixed State Convert
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
'''