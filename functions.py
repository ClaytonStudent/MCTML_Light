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