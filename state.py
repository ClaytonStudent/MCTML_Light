from search_space import data_actions,feature_actions,classifier_actions,check_list
import copy        

class State():
    def __init__(self,state=[],actions= data_actions):
        self.state = state
        self.actions = data_actions
        self.depth = 0
        
    def update_state(self,action):
        
        next_state = copy.deepcopy(self)
        
        if self.depth == 0:
            next_state.actions = feature_actions
            next_state.state.append(action)   
        elif self.depth == 1:
            next_state.actions = classifier_actions
            next_state.state.append(action) 
        elif self.depth in [2,3,4]:
            selected_method = next_state.state[next_state.depth-2]
            selected_method = check_list[selected_method]
            next_state.actions = selected_method.search_space
            next_state.state.append(action) 
        else:
            next_state.actions = None
            next_state.state.append(action)
            
        next_state.depth +=1 
        return next_state