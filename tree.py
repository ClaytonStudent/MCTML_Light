from state import State
import numpy as np
import random

class TreeNode(object):
    def __init__(self, parent,prior_p):
        self._name = None
        self._parent = parent
        self._children = {}  
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._state = State()
        
    #一次性全部expand
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self,prob) 
                self._children[action]._state = self._state.update_state(action)
                self._children[action]._name = action
                
    def add_new_node(self,action):
        if action not in self._children:
            self._children[action] = TreeNode(self,1.0)
            self._children[action]._state = self._state.update_state(action)
            self._children[action]._name = action
            #print('New Node added in the tree')
                            
    def select(self,c_puct):
        children_values = [i[1].get_value(c_puct) for i in self._children.items()]  
        selected_id = children_values.index(max(children_values))
        return list(self._children.items())[selected_id]
    
    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        
    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None
    
    
    
    
            
        
        