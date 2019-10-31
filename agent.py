from tree import TreeNode
from game import Store
from evaluator import get_score,name,policy_value_fn
from functions import game_end,has_continues,save_model,save_score

import pandas as pd
import time
import copy
import random
from operator import itemgetter

class MCTS(object):

    def __init__(self, total_time=100):
        self._policy = policy_value_fn
        self._c_puct = 2
        self._total_time = total_time
        self._storage = Store()
        self._root = TreeNode(None,1.0)

    def _playout(self):
        node = self._root
        while True:
            if node.is_leaf():
                break
            has_continues(node)
            action, node = node.select(self._c_puct)
        avaiables_actions = node._state.actions        
        action_probs = self._policy(avaiables_actions)    
        end = game_end(avaiables_actions)
        if not end:
            node.expand(action_probs)
        else:
            leaf_value = get_score(node)
            if leaf_value == -1:
                self.delete_one_pipeline(node)
            else:
                self._storage.add_one_record(node._state,leaf_value) 
                node.update_recursive(leaf_value)
                
    def _palyouts(self):
        start_time = time.time()
        end_time = start_time + self._total_time
        k=0
        while True:
            if time.time() > end_time:
                best_param,best_score = self.find_best()
                save_model(name,best_param,best_score)
                break
            else:
                n = int((time.time() - start_time) // 6)
                if n > k:
                    k= n
                    print('{}th min, best score is {}'.format(k,self.find_intermediate()))
                    save_score(k,self.find_intermediate())
                self._playout()
            
    def delete_one_pipeline(self,leaf_node):
        end_state = leaf_node._state.state
        self._root._children[end_state[0]]._children[end_state[1]]._children.pop(end_state[2],None)
        
    def find_intermediate(self):
        score = max([i[1] for i in self._storage.ps])
        return score
    
    def find_best(self):
        paths = self._storage.ps
        scores = [ps[1] for ps in paths]
        best = paths[scores.index(max(scores))]
        return best[0].state,best[1]
    

        
    
class Agent():
    """AI player based on MCTS"""
    def __init__(self, total_time=100):
        self.mcts = MCTS(total_time)
        
    def play_outs(self):
        self.mcts._palyouts()

