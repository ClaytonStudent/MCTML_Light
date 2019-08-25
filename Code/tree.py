from Code.state import State
#from Code.mct import MCT
from Code.evaluator import get_next_state
import numpy as np
import random

class TreeNode(object):
    def __init__(self, parent,name,prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._name = name
        self.state = State()
        
    #一次性全部expand
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self,action,prob) 
                self._children[action].state.id = get_next_state(self.state.id,action)
                self._children[action].state.update()
            
    def select(self,c_puct):
        children_values = [i[1].get_value(c_puct) for i in self._children.items()]
        m = max(children_values)
        id = [i for i, j in enumerate(children_values) if j == m]
        selected_id = random.choice(id)
        return list(self._children.items())[selected_id]

    def update(self, leaf_value):
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
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
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None
    
    def check_node_name(self):
        return type(self._name)==tuple
    
    def check_for_tuple(self):
        if list(self._children.values()) != []:
            if type(list(self._children.keys())[0]) == tuple:
                for key,value in self._children.items():
                    print("The visit time of {} is {}".format(key,value._n_visits))
    
            
        
        