from Code.mct import MCT
from Code.game import Game,Store
from Code.evaluator import update_parameter,pipeline_construction,pipeline_to_score
import time
#import copy
#import random


def plot_children_info(node):
    if node._children:
        children = list(node._children.values())
        for child in children:
            value = round(child._u+child._Q,2) 
            print("child id {}, visit time is {},value is {}".format(child.state.id,child._n_visits,value))

            
            
class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=3, n_playout=5,storage=Store()):
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._storage = storage

    def _playout(self, node, mct):
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
        avaiables_actions = Game.get_avaiable_actions(node.state)        
        action_probs = self._policy(avaiables_actions) 
        end = Game.game_end(node.state)
        if not end and avaiables_actions is not None:
            node.expand(action_probs)
            mct.add_all_nodes_to_tree(node)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(node)
        node.update_recursive(leaf_value)

    def _palyouts(self, node, mct):
        start_time = time.time()
        for n in range(self._n_playout):
            
            self._playout(node, mct)
            if n%10 == 9:
                print("The {}/{} playout".format(n+1,self._n_playout))
                plot_children_info(node)
        end_time = time.time()
        print("This playout time is",round(end_time-start_time,2))

    def _evaluate_rollout(self, node):
        # 随机采样其他的值
        parameter = update_parameter(node.state)
        #print("Sampled Parameter:",parameter)
        pipe = pipeline_construction(parameter)
        score = pipeline_to_score(pipe)
        #print("Score:",score)
        #score = 1
        self._storage.add_one_record(parameter,score)
        return score

class Agent():
    """AI player based on MCTS"""
    def __init__(self,policy_value_fn, c_puct=10, n_playout=50):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.mct = MCT()

    def reset_mct(self):
        pass
    
    def episode(self):
        pass
        #self.mcts._playout(self.mct)
        
    def play_outs(self):
        node = self.mct.root
        self.mcts._palyouts(node,self.mct)
    
    def one_playout(self):
        node = self.mct.root
        self.mcts._playout(node,self.mct)
        