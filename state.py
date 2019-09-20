from Basic import name_parameter,name_model,names
import random

def param_to_dics(params):
    actions = []
    for i,values in params.items():
        for value in values:
            actions.append((i,value))
    return actions

class State:
    def __init__(self,path=[],actions=names,level=0,hypername=None):
        self.path = path
        self.actions = actions
        self.level = level
    
    def update_action(self,action):
        if self.level == 0:
            actions = param_to_dics(name_parameter[action])
        elif self.level >= 4 or random.random()>0.9:
            actions = None 
        else:
            actions = param_to_dics(name_parameter[self.path[0]])
        return actions
    
    def update(self,action):
        
        path = self.path + [action]
        actions = self.update_action(action)
        level = self.level + 1
        return State(path=path,actions=actions,level=level)
        
    def update_self(self,action):
        self.path += [action]
        self.actions = self.update_action(action)
        self.level += 1

'''

class State:
    def __init__(self,path=[],actions=names,level=0):
        self.path = path
        self.actions = actions
        self.level = level
    
    def update_action(self,action):
        if self.level == 0:
            model_name = action
            model_parameter = name_parameter[model_name]
            actions = model_parameter[list(model_parameter.keys())[0]]
        elif self.level >= len(name_parameter[self.path[0]]):
            actions = None
        else:
            model_name = self.path[0]
            model_parameter = name_parameter[model_name]
            actions = model_parameter[list(model_parameter.keys())[self.level]]
        return actions
    
    def update(self,action):
        
        path = self.path + [action]
        actions = self.update_action(action)
        level = self.level + 1
        return State(path=path,actions=actions,level=level)
        
    def update_self(self,action):
        self.path += [action]
        self.actions = self.update_action(action)
        self.level += 1
'''   