from mct import MCT
from tree import TreeNode
from game import Game,Store
import time
import copy
import random
from operator import itemgetter
from evaluator import rollout_policy_fn,get_score
from functions import extract_training_data
from network import run_nn

            
class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=3, n_playout=5):
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._storage = Store()
        self._root = TreeNode(None,None,1.0)
        self._mct = MCT(self._root)


    def _playout(self):
        node = self._root
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
        avaiables_actions = Game.get_avaiable_actions(node.state)        
        action_probs = self._policy(avaiables_actions) 
        end = Game.game_end(node.state)
        if not end and avaiables_actions is not None:
            node.expand(action_probs)
            self._mct.add_all_nodes_to_tree(node)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(copy.deepcopy(node))
        node.update_recursive(leaf_value)

    def _palyouts(self):
        for n in range(self._n_playout):
            print("Playout: ",n+1)
            self._playout()
            
            if n%21 == 19:
                X,y = extract_training_data(self._storage.ps,self._mct.nodes)
                run_nn(X,y,'save/model.h5')
                
          
            
    def _evaluate_rollout(self, node):
        # 随机采样其他的值
        for i in range(15):
            end = Game.game_end(node.state)
            if end:
                score = get_score(node.state.path)
                break
            action_probs = rollout_policy_fn(node.state.actions)
            max_action = max(action_probs, key=itemgetter(1))[0]
            node.state.update_self(max_action)
        else:
            score = 0
        self._storage.add_one_record(node.state.path,score)
        return score
    
class Agent():
    """AI player based on MCTS"""
    def __init__(self,policy_value_fn, c_puct=15, n_playout=10):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        
    def play_outs(self):
        self.mcts._palyouts()

        