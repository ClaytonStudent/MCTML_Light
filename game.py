from evaluator import X,y
from sklearn.model_selection import ShuffleSplit,cross_val_score
import random

class Store():
    def __init__(self):
        self.parameter = []
        self.score = []
        self.ps = []
    
    def add_one_record(self,parameter,score):
        self.parameter.append(parameter)
        self.score.append(score)
        self.ps.append((parameter,score))
        
    def get_score_from_history(self,parameter):
        index = self.parameter.index(parameter)
        return self.score[index]
    
    def check_if_useless(parameter,score):
        pass
        
    

class Game():
    def __init__(self):
        pass
    
    def start(self):
        return search_space
    
    @staticmethod
    def get_avaiable_actions(state):
        if Game.game_end(state):
            return None
        else:
            return state.actions
    
    @staticmethod
    def game_end(state):
        if state.actions == None:
            return True
        else:
            return False
    
   
        

        