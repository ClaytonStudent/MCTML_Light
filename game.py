from evaluator import X,y
#from setup import search_space,search_space_dic
from sklearn.model_selection import ShuffleSplit,cross_val_score
import random
#from collections import Counter



class Store():
    def __init__(self):
        self.parameter = []
        self.score = []
        self.ps = []
    
    def add_one_record(self,parameter,score):
        self.parameter.append(parameter)
        self.score.append(score)
        self.ps.append((parameter,score))
    

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
    
    
    @staticmethod
    def get_score(parameter):
        pipe_list = parameter_to_pipeline(parameter,search_space_dic)
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        score = cross_val_score(pipe_list,X,y,cv=cv)
        return score.mean()  
        

        