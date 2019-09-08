from mct import MCT
from game import Game,Store
from evaluator import update_parameter,pipeline_construction,pipeline_to_score
import time
import copy
import random
from operator import itemgetter
from evaluator import rollout_policy_fn,get_score_with_one_feature,get_score

def plot_children_info(node):
    if node._children:
        children = list(node._children.values())
        n_visit_times = [child._n_visits for child in children]
        Q_values = [child._Q for child in children]
        for child in children:
            value = round(child._u+child._Q,2) 
            #print("child id {}, visit time is {},value is {}".format(child.state.path,child._n_visits,value)) 
        return n_visit_times,Q_values
    else:
        print("No children values ")
            
class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=3, n_playout=5):
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._storage = Store()

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
        leaf_value = self._evaluate_rollout(copy.deepcopy(node))
        node.update_recursive(leaf_value)

    def _palyouts(self, node, mct):
        times = []
        visit_times = []
        Q_values = []
        
        start_time = time.time()
        for n in range(self._n_playout):
            self._playout(node, mct)
            visit_time,q_value = plot_children_info(node)
            visit_times.append(visit_time)
            Q_values.append(q_value)
            used_time = time.time() - start_time
            times.append(used_time)
        return times,visit_times,Q_values

    def _evaluate_rollout(self, node):
        # 随机采样其他的值
        for i in range(15):
            end = Game.game_end(node.state)
            if end:
                score = get_score(node.state.path)
                #score = get_score_with_one_feature(node.state.path)
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
        self.mct = MCT()
        
    def play_outs(self):
        node = self.mct.root
        times,visit_times,Q_values = self.mcts._palyouts(node,self.mct)
        return times,visit_times,Q_values
        