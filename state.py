#from Code.setup import search_space_dic,search_space
from Basic import name_parameter,name_model,names


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
        